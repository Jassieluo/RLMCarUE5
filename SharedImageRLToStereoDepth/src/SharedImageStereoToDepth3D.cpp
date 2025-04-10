#include <Windows.h>
#include <opencv2/opencv.hpp>
#include "opencv2/viz.hpp"

LPCWSTR SharedMemoryNameDisp = L"CameraDispSharedMemory";
const int SharedMemorySizeDisp = 512 * 256 * 1;
HANDLE hMapFileDisp;
LPVOID pBufferDisp;

LPCWSTR SharedMemoryNameCommand = L"CommandSharedMemory";
const int SharedMemorySizeCommand = 1+1+1;
HANDLE hMapFileCommand;
LPVOID pBufferCommand;

int CameraWidth = 512;
int CameraHeight = 256;
float Pi = 3.1415926;
int cx = CameraWidth / 2;
int cy = CameraHeight / 2;
float FOV = 120.f * (Pi / 180.0f);
float f = CameraWidth / (2 * tan(FOV / 2.f));
float T = 0.25;

cv::Mat Q = (cv::Mat_<double>(4, 4) <<
    1, 0, 0, -cx, // cx 为主点 x 坐标
    0, 1, 0, -cy, // cy 为主点 y 坐标
    0, 0, 0, f,   // f 为焦距
    0, 0, -1 / T, 0  // T 为基线距离
    );

float* processPointClouds(cv::Mat PointCloud) {
    cv::Mat pointCloud = PointCloud;

    // 初始化避障参数
    const float minSafeDistance = 1.5f; // 最小安全距离（单位：毫米）
    const float maxSpeed = 1.0f; // 最大速度
    const float maxLateralSpeed = 1.0f; // 最大侧向速度
    const float maxRotationSpeed = 1.0f; // 最大旋转速度

    float* controlCommand = new float[3];

    // 分区域检测点云数据
    // 定义感兴趣区域（ROI）以检测障碍物
    int roiLeft = pointCloud.cols / 4; // ROI的左右边界
    int roiRight = 3 * pointCloud.cols / 4;
    int roiTop = 0; // ROI的上下边界（考虑全范围）
    int roiBottom = pointCloud.rows;

    // 统计障碍物点数量
    int obstacleCountR = 0;
    int obstacleCountL = 0;
    int totalPoints = 0;

    for (int y = roiTop; y < roiBottom; y++) {
        for (int x = roiLeft; x < roiRight; x++) {
            cv::Vec3f point = pointCloud.at<cv::Vec3f>(y, x);
            float distance = -point[2]; // Z坐标作为深度
            //if (y >= 85 && y <= 170 && x >= 170 && x <= 340)
            //printf("Distance: %f\n", distance);
            if (y >= 70 && y <= 180 && x >= 256 && x <= 376 && distance < minSafeDistance && distance > 0) {
                obstacleCountR++;
            }
            if (y >= 70 && y <= 180 && x >= 136 && x <= 256 && distance < minSafeDistance && distance > 0) {
                obstacleCountL++;
            }
            totalPoints++;
        }
    }

    float obstacleRatioR = static_cast<float>(obstacleCountR) / totalPoints;
    float obstacleRatioL = static_cast<float>(obstacleCountL) / totalPoints;
    // 根据检测结果生成控制指令
    if (obstacleRatioL > 0.1) { // 障碍物占比阈值
        // 障碍物较多，减速并避让
        controlCommand[0] = 0.5f * maxSpeed; // 减速
        controlCommand[1] = -0.5f * maxLateralSpeed; // 向左避让
        controlCommand[2] = -maxRotationSpeed; // 逆时针旋转
    }
    else if(obstacleRatioR > 0.1){
        // 无障碍物，正常行驶
        controlCommand[0] = maxSpeed;
        controlCommand[1] = 0.5f * maxLateralSpeed;
        controlCommand[2] = maxRotationSpeed;
    }
    else
    {
        controlCommand[0] = maxSpeed;
        controlCommand[1] = 0;
        controlCommand[2] = 0;
    }
    return controlCommand;
}


int main()
{
    hMapFileDisp = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        SharedMemoryNameDisp
    );
    printf("%d\n", hMapFileDisp);
    if (hMapFileDisp == NULL) {
        printf("OpenFileMapping failed\n");
        return 1;
    }
    pBufferDisp = MapViewOfFile(
        hMapFileDisp,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        SharedMemorySizeDisp * sizeof(float)
    );
    if (pBufferDisp == NULL) {
        printf("MapViewOfFile failed\n");
        CloseHandle(hMapFileDisp);
        return 1;
    }

    hMapFileCommand = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(float) * SharedMemorySizeCommand,
        SharedMemoryNameCommand
    );
    if (hMapFileCommand == NULL) {
        printf("OpenFileMapping failed\n");
        return 1;
    }
    pBufferCommand = MapViewOfFile(
        hMapFileCommand,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        SharedMemorySizeCommand * sizeof(float)
    );
    if (pBufferCommand == NULL) {
        printf("MapViewOfFile failed\n");
        CloseHandle(hMapFileCommand);
        return 1;
    }

    int height = 255;
    int width = 512;
    int type = CV_32FC1;
    float* dataDisp = (float*)malloc(sizeof(float) * width * height * 1);
    cv::Mat DepthPred;
    cv::viz::Viz3d CloudWindow("window");
    CloudWindow.setBackgroundColor();
    while (cv::waitKey(1) != 27)
    {
        memcpy(dataDisp, (const void*)(pBufferDisp), sizeof(float)* width* height * 1);
        cv::Mat ImageDisp(height, width, type, dataDisp);
        cv::Mat disp_flip;
        cv::flip(ImageDisp, disp_flip, 1);
        cv::reprojectImageTo3D(disp_flip, DepthPred, Q, false);
        
        cv::viz::WCloud cloud(DepthPred, cv::viz::Viz3d::Color::white()); // 显示点云和颜色

        memcpy((void*)(pBufferCommand), processPointClouds(DepthPred), sizeof(float) * 3);

        CloudWindow.showWidget("cloud", cloud);
        CloudWindow.showWidget("Coordinate", cv::viz::WCoordinateSystem()); // 显示坐标

        cv::viz::WCube cube_widget(cv::Point3f(-0.25, -0.55, -0.3), cv::Point3f(0.25, -0.15, 0.7), true, cv::viz::Color::red());

        cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0); // 设置线条粗细

        CloudWindow.showWidget("Cube Widget", cube_widget);
        CloudWindow.spinOnce(1, true);
    }
    free(dataDisp);
    UnmapViewOfFile(pBufferDisp);
    CloseHandle(hMapFileDisp);
    UnmapViewOfFile(pBufferCommand);
    CloseHandle(hMapFileCommand);
}