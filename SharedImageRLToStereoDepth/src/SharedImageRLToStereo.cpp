#include <Windows.h>
#include "process_utils.h"
#include "utils.h";

int CameraWidth = 512;
int CameraHeight = 256;
float Pi = 3.1415926;
int cx = CameraWidth / 2;
int cy = CameraHeight / 2;
float FOV = 120.f * (Pi / 180.0f);
float f = CameraWidth / (2 * tan(FOV / 2.f));
float T = 0.25;
double baseLine = 0.27;

cv::Mat Q = (cv::Mat_<double>(4, 4) <<
    1, 0, 0, -cx, // cx 为主点 x 坐标
    0, 1, 0, -cy, // cy 为主点 y 坐标
    0, 0, 0, f,   // f 为焦距
    0, 0, -1 / T, 0  // T 为基线距离
    );

LPCWSTR SharedMemoryNameL = L"CameraLRGBASharedMemory";
const int SharedMemorySizeL = 512 * 256 * 3;
HANDLE hMapFileL;
LPVOID pBufferL;

LPCWSTR SharedMemoryNameR = L"CameraRRGBASharedMemory";
const int SharedMemorySizeR = 512 * 256 * 3;
HANDLE hMapFileR;
LPVOID pBufferR;

LPCWSTR SharedMemoryNameDisp = L"CameraDispSharedMemory";
const int SharedMemorySizeDisp = 512 * 256 * 1;
HANDLE hMapFileDisp;
LPVOID pBufferDisp;

LPCWSTR SharedMemoryNameDepth = L"CameraDepthSharedMemory";
const int SharedMemorySizeDepth = 512 * 256 * 1;
HANDLE hMapFileDepth;
LPVOID pBufferDepth;

int nGpuId = 1;
cudaError_t cudaStatus = cudaSetDevice(nGpuId);
//if (cudaStatus != cudaSuccess) {
//    fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
//    return 1;
//}

std::string engine_path = "E:\\exprements_pycharm\\SharedImageRLToStereoDepth\\weights\\LightStereo-S-SceneFlow_fp16.engine";

nvinfer1::IRuntime* runtime;
nvinfer1::ICudaEngine* engine;
nvinfer1::IExecutionContext* context;

float* device_buffers[3];
uint8_t* image_l_device = nullptr;
uint8_t* image_r_device = nullptr;

int input_img_l_height;
int input_img_l_weight;
int input_img_r_height;
int input_img_r_weight;
int output_img_height;
int output_img_weight;

float* output_buffer_host = new float[3 * output_img_height * output_img_weight];

uint8_t* image_device = nullptr;

cv::Mat disparityToDepth(const cv::Mat& disparity, double focalLength, double baseline) {
    cv::Mat depth = disparity.clone();
    // 避免除以0
    depth.setTo(1e-5, depth == 0);
    depth = focalLength * baseline / depth;
    return depth;
}

int main()
{
    hMapFileL = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        SharedMemoryNameL
    );

    hMapFileR = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        SharedMemoryNameR
    );

    if (hMapFileL == NULL || hMapFileR == NULL) {
        printf("OpenFileMapping failed\n");
        return 1;
    }

    pBufferL = MapViewOfFile(
        hMapFileL,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        SharedMemorySizeL * sizeof(uchar)
    );
    pBufferR = MapViewOfFile(
        hMapFileR,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        SharedMemorySizeR * sizeof(uchar)
    );

    if (pBufferL == NULL || pBufferR == NULL) {
        printf("MapViewOfFile failed\n");
        CloseHandle(hMapFileL);
        CloseHandle(hMapFileR);
        return 1;
    }

    hMapFileDisp = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(float) * SharedMemorySizeDisp,
        SharedMemoryNameDisp
    );
    if (hMapFileDisp == NULL) {
        printf("CreateFileDepthMapping failed\n");
        return 1;
    }
    pBufferDisp = MapViewOfFile(
        hMapFileDisp,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(float) * SharedMemorySizeDisp
    );
    if (pBufferDisp == NULL) {
        printf("MapViewDispOfFile failed\n");
        CloseHandle(hMapFileDisp);
        return 1;
    }

    hMapFileDepth = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(float) * SharedMemorySizeDepth,
        SharedMemoryNameDepth
    );
    if (hMapFileDepth == NULL) {
        printf("CreateFileDepthMapping failed\n");
        return 1;
    }
    pBufferDepth = MapViewOfFile(
        hMapFileDepth,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(float) * SharedMemorySizeDepth
    );
    if (pBufferDepth == NULL) {
        printf("MapViewDepthOfFile failed\n");
        CloseHandle(hMapFileDepth);
        return 1;
    }

    int height = 256;
    int width = 512;
    int type = CV_8UC3;
    uchar* dataL = (uchar*)malloc(sizeof(uchar) * width * height * 3);
    uchar* dataR = (uchar*)malloc(sizeof(uchar) * width * height * 3);


    readEngineFile(engine_path, runtime, engine, context);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    input_img_l_height = engine->getTensorShape(engine->getIOTensorName(0)).d[engine->getTensorShape(engine->getIOTensorName(0)).nbDims - 2];
    input_img_l_weight = engine->getTensorShape(engine->getIOTensorName(0)).d[engine->getTensorShape(engine->getIOTensorName(0)).nbDims - 1];
    input_img_r_height = engine->getTensorShape(engine->getIOTensorName(1)).d[engine->getTensorShape(engine->getIOTensorName(1)).nbDims - 2];
    input_img_r_weight = engine->getTensorShape(engine->getIOTensorName(1)).d[engine->getTensorShape(engine->getIOTensorName(1)).nbDims - 1];
    output_img_height = engine->getTensorShape(engine->getIOTensorName(2)).d[engine->getTensorShape(engine->getIOTensorName(2)).nbDims - 2];
    output_img_weight = engine->getTensorShape(engine->getIOTensorName(2)).d[engine->getTensorShape(engine->getIOTensorName(2)).nbDims - 1];

    cudaMalloc((void**)&image_device, 3 * input_img_l_height * input_img_l_weight);
    // cudaMalloc((void**)&image_device, 3 * capture_height * capture_width);
    cudaMalloc((void**)&device_buffers[0], sizeof(float) * 3 * input_img_l_height * input_img_l_weight);
    cudaMalloc((void**)&device_buffers[1], sizeof(float) * 3 * input_img_r_height * input_img_r_weight);
    cudaMalloc((void**)&device_buffers[2], sizeof(float) * 1 * output_img_height * output_img_weight);

    context->setTensorAddress(engine->getIOTensorName(0), device_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), device_buffers[1]);
    context->setTensorAddress(engine->getIOTensorName(2), device_buffers[2]);

    while (cv::waitKey(1)!=27)
    {
        cv::Mat disp_pred(cv::Size(output_img_weight, output_img_height), CV_32FC1);
        cv::Mat normalized_disp_pred;
        cv::Mat color_normalized_disp_pred;
        cv::Mat DepthPred;

        memcpy(dataL, (const void*)(pBufferL), sizeof(uchar) * width * height * 3);
        memcpy(dataR, (const void*)(pBufferR), sizeof(uchar) * width * height * 3);

        cv::Mat ImageL(height, width, type, dataL);
        cv::cvtColor(ImageL, ImageL, cv::COLOR_RGB2BGR);
        ImageL.convertTo(ImageL, ImageL.type(), 1.5, 10);
        cv::Mat ImageR(height, width, type, dataR);
        cv::cvtColor(ImageR, ImageR, cv::COLOR_RGB2BGR);
        ImageR.convertTo(ImageR, ImageR.type(), 1.5, 10);

        auto t_beg = std::chrono::high_resolution_clock::now();

        cudaMemcpyAsync(image_device, ImageL.data, ImageL.size().area() * 3, cudaMemcpyHostToDevice, stream);
        preprocess_no_resize(image_device, ImageL.size().width, ImageL.size().height, device_buffers[0], input_img_l_weight, input_img_l_height, stream);
        cudaMemcpyAsync(image_device, ImageR.data, ImageR.size().area() * 3, cudaMemcpyHostToDevice, stream);
        preprocess_no_resize(image_device, ImageR.size().width, ImageR.size().height, device_buffers[1], input_img_r_weight, input_img_r_height, stream);

        context->enqueueV3(stream);

        cudaMemcpyAsync(disp_pred.data, device_buffers[2], output_img_height * output_img_weight * sizeof(float), cudaMemcpyDeviceToHost, stream);

        auto t_end = std::chrono::high_resolution_clock::now();
        float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        
        DepthPred = disparityToDepth(disp_pred, f, baseLine);

        CopyMemory(pBufferDisp, disp_pred.data, sizeof(float) * SharedMemorySizeDisp);
        CopyMemory(pBufferDepth, DepthPred.data, sizeof(float) * SharedMemorySizeDepth);

        std::cout << "Inference time: " << int(total_inf) << std::endl;
        //double minVal, maxVal;
        //cv::minMaxLoc(disp_pred, &minVal, &maxVal);
        //disp_pred.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        //cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
        //cv::imshow("color_normalized_disp_pred", color_normalized_disp_pred);
        

        //cv::imshow("ImageL", ImageL);
        //cv::imshow("ImageR", ImageR);
    }
    
    UnmapViewOfFile(pBufferL);
    CloseHandle(hMapFileL);
    UnmapViewOfFile(pBufferR);
    CloseHandle(hMapFileR);
    UnmapViewOfFile(pBufferDisp);
    CloseHandle(hMapFileDisp);
    UnmapViewOfFile(pBufferDepth);
    CloseHandle(hMapFileDepth);
    //UnmapViewOfFile(pBufferDepth);
    //CloseHandle(hMapFileDepth);
    cv::destroyAllWindows();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    cudaFree(device_buffers[2]);
    delete[] output_buffer_host;
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
    free(dataL);
    free(dataR);
    return 0;
}