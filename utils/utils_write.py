from utils_ctypes_winapi import CreateFileMappingW, MapViewOfFile, FILE_MAP_WRITE, INVALID_HANDLE_VALUE, PAGE_READWRITE
import ctypes


class GameResetSharedTools:
    def __init__(self, SharedResetGameName="ResetGameSharedMemory", SharedResetGameSize=1, ResetGameCommand=1):
        self.SharedResetGameName = SharedResetGameName
        self.SharedResetGameSize = SharedResetGameSize
        self.ResetGameCommand = ResetGameCommand
        self.h_map_reset_game = None
        self.ptr_reset_game = None

    def create_shared_memory(self):
        self.h_map_reset_game = CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, self.SharedResetGameSize * 4, self.SharedResetGameName)
        self.ptr_reset_game = MapViewOfFile(self.h_map_reset_game, FILE_MAP_WRITE, 0, 0, self.SharedResetGameSize * 4) if self.h_map_reset_game else None

    def reset_game(self):
        if self.ptr_reset_game:
            ctypes.cast(self.ptr_reset_game, ctypes.POINTER(ctypes.c_int)).contents.value = int(self.ResetGameCommand)
        else:
            print("failed to reset game, ptr_reset_game is null!")