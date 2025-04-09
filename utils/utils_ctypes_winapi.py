import ctypes
from ctypes import wintypes

FILE_MAP_READ = 0x0004
FILE_MAP_WRITE = 0x0002

INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
PAGE_READWRITE = 0x04

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenFileMappingW.restype = wintypes.HANDLE

CreateFileMappingW = kernel32.CreateFileMappingW
CreateFileMappingW.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR]

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = ctypes.c_void_p