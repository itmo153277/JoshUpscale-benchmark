include(FindPackageHandleStandardArgs)
find_path(onnxruntime_INCLUDE_DIR onnxruntime_c_api.h PATH_SUFFIXES include)
find_library(onnxruntime_LIBRARY onnxruntime PATH_SUFFIXES lib lib64 lib/x64)
find_library(onnxruntime_LIBRARY_CUDA onnxruntime_providers_cuda PATH_SUFFIXES lib lib64 lib/x64)
find_library(onnxruntime_LIBRARY_TENSORRT onnxruntime_providers_tensorrt PATH_SUFFIXES lib lib64 lib/x64)
find_package_handle_standard_args(
  onnxruntime DEFAULT_MSG
  onnxruntime_INCLUDE_DIR
  onnxruntime_LIBRARY
)
mark_as_advanced(
  onnxruntime_INCLUDE_DIR
  onnxruntime_LIBRARY
)
if (onnxruntime_FOUND)
  add_library(onnxruntime SHARED IMPORTED)
  set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}")
  if (WIN32)
    set_property(TARGET onnxruntime PROPERTY IMPORTED_IMPLIB "${onnxruntime_LIBRARY}")
  else()
    set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
  endif()
  if (onnxruntime_LIBRARY_CUDA)
    add_library(onnxruntime::cuda SHARED IMPORTED)
    if (WIN32)
      set_property(TARGET onnxruntime::cuda PROPERTY IMPORTED_IMPLIB "${onnxruntime_LIBRARY_CUDA}")
    else()
      set_property(TARGET onnxruntime::cuda PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY_CUDA}")
    endif()
  endif()
  if (onnxruntime_LIBRARY_TENSORRT)
    add_library(onnxruntime::tensorrt SHARED IMPORTED)
    if (WIN32)
      set_property(TARGET onnxruntime::tensorrt PROPERTY IMPORTED_IMPLIB "${onnxruntime_LIBRARY_TENSORRT}")
    else()
      set_property(TARGET onnxruntime::tensorrt PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY_TENSORRT}")
    endif()
  endif()
endif()
