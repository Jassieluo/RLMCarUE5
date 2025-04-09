import ctypes
import torch
from utils_ctypes_winapi import OpenFileMappingW, MapViewOfFile, FILE_MAP_READ
import numpy as np


class SharedUETrainMessageReadTools:
    def __init__(self, DispName=r"CameraDispSharedMemory", DispSize=512 * 256 * 1,
                     DepthName=r"CameraDepthSharedMemory", DepthSize=512 * 256 * 1,
                     CameraLName=r"CameraLRGBASharedMemory", CameraLSize=512 * 256 * 3,
                     CameraRName=r"CameraRRGBASharedMemory", CameraRSize=512 * 256 * 3,
                     VectorAndCollisionName=r"VectorAndCollisionSharedMemory", vectorAndCollisionSize=2 + 2 + 2 + 1, epsilon=1e-6,
                     device='cuda:0'):

        self.DispName = DispName
        self.DispSize = DispSize
        self.h_map_disp = None
        self.ptr_disp = None

        self.DepthName = DepthName
        self.DepthSize = DepthSize
        self.h_map_depth = None
        self.ptr_depth = None

        self.CameraLName = CameraLName
        self.CameraLSize = CameraLSize
        self.h_map_camera_l = None
        self.ptr_camera_l = None

        self.CameraRName = CameraRName
        self.CameraRSize = CameraRSize
        self.h_map_camera_r = None
        self.ptr_camera_r = None

        self.VectorAndCollisionName = VectorAndCollisionName
        self.vectorAndCollisionSize = vectorAndCollisionSize
        self.h_map_vectorAndCollision = None
        self.ptr_vectorAndCollision = None

        self.epsilon = epsilon

        self.device = device

    def create_disp_recv_map(self):
        self.h_map_disp = OpenFileMappingW(FILE_MAP_READ, False, self.DispName)
        self.ptr_disp = MapViewOfFile(self.h_map_disp, FILE_MAP_READ, 0, 0, self.DispSize * 4) if self.h_map_disp else None
        if not self.ptr_disp:
            print("Create ptr_disp failed!")
            return False
        return True

    def create_depth_recv_map(self):
        self.h_map_depth = OpenFileMappingW(FILE_MAP_READ, False, self.DepthName)
        self.ptr_depth = MapViewOfFile(self.h_map_depth, FILE_MAP_READ, 0, 0, self.DepthSize * 4) if self.h_map_depth else None
        if not self.ptr_depth:
            print("Create ptr_depth failed!")
            return False
        return True

    def create_camera_l_recv_map(self):
        self.h_map_camera_l = OpenFileMappingW(FILE_MAP_READ, False, self.CameraLName)
        self.ptr_camera_l = MapViewOfFile(self.h_map_camera_l, FILE_MAP_READ, 0, 0, self.CameraLSize) if self.h_map_camera_l else None
        if not self.ptr_camera_l:
            print("Create ptr_camera_l failed!")
            return False
        return True

    def create_camera_r_recv_map(self):
        self.h_map_camera_r = OpenFileMappingW(FILE_MAP_READ, False, self.CameraRName)
        self.ptr_camera_r = MapViewOfFile(self.h_map_camera_r, FILE_MAP_READ, 0, 0, self.CameraRSize) if self.h_map_camera_r else None
        if not self.ptr_camera_r:
            print("Create ptr_camera_r failed!")
            return False
        return True

    def create_vector_recv_map(self):
        self.h_map_vectorAndCollision = OpenFileMappingW(FILE_MAP_READ, False, self.VectorAndCollisionName)
        self.ptr_vectorAndCollision = MapViewOfFile(self.h_map_vectorAndCollision, FILE_MAP_READ, 0, 0, self.vectorAndCollisionSize * 4) if self.h_map_vectorAndCollision else None
        if not self.ptr_vectorAndCollision:
            print("Create ptr_vectorAndCollision failed!")
            return False
        return True

    def get_shared_disp_tensor(self):
        return torch.tensor(ctypes.cast(self.ptr_disp, ctypes.POINTER(ctypes.c_float * self.DispSize)).contents, dtype=torch.float).reshape(
            [1, 1, 256, 512]).to(self.device) if self.ptr_disp else None

    def get_shared_disp_norm_log_tensor(self):
        disp_log = np.log(self.get_shared_disp_np() + self.epsilon)
        min_log_disp = np.min(disp_log)
        max_log_disp = np.max(disp_log)
        return torch.tensor((disp_log - min_log_disp) / (max_log_disp - min_log_disp), dtype=torch.float).reshape(
            [1, 1, 256, 512]).to(self.device) if self.ptr_disp else None

    def get_shared_disp_norm_linear_tensor(self):
        disp_np = self.get_shared_disp_np()
        min_disp = np.min(disp_np)
        max_disp = np.max(disp_np)
        return torch.tensor((disp_np - min_disp) / (max_disp - min_disp), dtype=torch.float).reshape(
            [1, 1, 256, 512]).to(self.device) if self.ptr_disp else None

    def get_shared_disp_np(self):
        return np.reshape(np.array(ctypes.cast(self.ptr_disp, ctypes.POINTER(ctypes.c_float * self.DispSize)).contents, dtype=np.float32), [1, 256, 512])\
            if self.ptr_disp else None

    def get_shared_disp_np_linear_norm(self):
        disp_np = self.get_shared_disp_np()
        min_disp = np.min(disp_np)
        max_disp = np.max(disp_np)
        return (disp_np - min_disp) / (max_disp - min_disp)

    def get_shared_depth_tensor(self):
        return torch.tensor(ctypes.cast(self.ptr_depth, ctypes.POINTER(ctypes.c_float * self.DepthSize)).contents, dtype=torch.float).reshape(
            [1, 1, 256, 512]).to(self.device) if self.ptr_depth else None

    def get_shared_depth_np(self):
        return np.reshape(np.array(ctypes.cast(self.ptr_depth, ctypes.POINTER(ctypes.c_float * self.DepthSize)).contents), [1, 256, 512])\
            if self.ptr_depth else None

    def get_shared_camera_l_norm_np(self):
        return np.reshape(np.array(ctypes.cast(self.ptr_camera_l, ctypes.POINTER(ctypes.c_int8 * self.CameraLSize)).contents, dtype=float), [3, 256, 512])/255.0\
            if self.ptr_camera_l else None

    def get_shared_camera_l_norm_tensor(self):
        return torch.tensor(np.reshape(np.array(ctypes.cast(self.ptr_camera_l, ctypes.POINTER(ctypes.c_int8 * self.CameraLSize)).contents, dtype=float), [1, 3, 256, 512])/255.0, dtype=torch.float).to(self.device)\
            if self.ptr_camera_l else None

    def get_shared_camera_r_norm_np(self):
        return np.reshape(np.array(ctypes.cast(self.ptr_camera_r, ctypes.POINTER(ctypes.c_int8 * self.CameraRSize)).contents, dtype=float), [3, 256, 512])/255.0\
            if self.ptr_camera_r else None

    def get_shared_camera_r_norm_tensor(self):
        return torch.tensor(np.reshape(np.array(ctypes.cast(self.ptr_camera_r, ctypes.POINTER(ctypes.c_int8 * self.CameraRSize)).contents, dtype=float), [1, 3, 256, 512])/255.0).to(self.device)\
            if self.ptr_camera_r else None

    def get_shared_vector_and_collision_list(self):
        return list(ctypes.cast(self.ptr_vectorAndCollision, ctypes.POINTER(ctypes.c_float * self.vectorAndCollisionSize)).contents)\
            if self.ptr_vectorAndCollision else None

    def get_shared_camera_l_disp_norm_np_bgrd(self):
        if not self.ptr_camera_l:
            print("ptr_camera_l is NULL!")
            return None
        if not self.ptr_disp:
            print("ptr_disp is NULL!")
            return None
        disp_log = np.log(self.get_shared_disp_np() + self.epsilon)
        min_log_disp = np.min(disp_log)
        max_log_disp = np.max(disp_log)
        return np.vstack([(disp_log - min_log_disp) / (max_log_disp - min_log_disp), self.get_shared_camera_l_norm_np()])

    def get_shared_camera_l_disp_norm_tensor_bgrd(self):
        return torch.tensor(self.get_shared_camera_l_disp_norm_np_bgrd(), dtype=torch.float).reshape([1, 4, 256, 512]).to(self.device)


class SharedCommandMessageReadTools:
    def __init__(self, CommandName="CommandSharedMemory", CommandSize=1+1+1):
        self.CommandName = CommandName
        self.CommandSize = CommandSize
        self.h_map_Command = None
        self.ptr_Command = None

    def create_command_recv_map(self):
        self.h_map_Command = OpenFileMappingW(FILE_MAP_READ, False, self.CommandName)
        self.ptr_Command = MapViewOfFile(self.h_map_Command, FILE_MAP_READ, 0, 0, self.CommandSize * 4) if self.h_map_Command else None
        if not self.ptr_Command:
            print("Create ptr_Command failed!")
            return False
        return True

    def get_command_list(self):
        return list(ctypes.cast(self.ptr_Command, ctypes.POINTER(ctypes.c_float * self.CommandSize)).contents) if self.ptr_Command else None