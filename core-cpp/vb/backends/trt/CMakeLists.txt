# Jetson TensorRT 10.x backend (BASE-1 M2.2).
# Sources are intentionally an OBJECT library so register.cpp is retained.
set(TRT_ROOT "/usr" CACHE PATH "TensorRT installation root")
enable_language(CUDA)
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS "${TRT_ROOT}/include" /usr/include/aarch64-linux-gnu REQUIRED)
find_library(TRT_LIBRARY nvinfer HINTS "${TRT_ROOT}/lib" /usr/lib/aarch64-linux-gnu REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(vb_trt OBJECT trt_backend.cpp trt_runner.cpp trt_stage2.cpp trt_source.cpp preprocess_cuda.cu
    ../rknn/model_sha256.cpp register.cpp)
target_include_directories(vb_trt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TRT_INCLUDE_DIR}
    $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(vb_trt PRIVATE ${TRT_LIBRARY} CUDA::cudart)
target_compile_definitions(vb_trt PRIVATE
    $<TARGET_PROPERTY:vb_core,INTERFACE_COMPILE_DEFINITIONS>)
set_target_properties(vb_trt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT VB_WITH_GST)
    message(FATAL_ERROR "vb: trt backend needs -DVB_WITH_GST=ON (NVDEC source)")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_TRT REQUIRED IMPORTED_TARGET
    gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
target_link_libraries(vb_trt PRIVATE PkgConfig::GST_TRT)
target_link_libraries(vb-runtime PRIVATE vb_trt)
message(STATUS "vb: trt backend enabled (TensorRT: ${TRT_LIBRARY})")

if(VB_TESTS)
    add_executable(vb_trt_reference_test trt_reference_test.cpp)
    target_include_directories(vb_trt_reference_test PRIVATE
        $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(vb_trt_reference_test PRIVATE vb_core)
    add_test(NAME vb_trt_reference_test COMMAND vb_trt_reference_test)
    add_executable(vb_trt_cuda_reference_test trt_cuda_reference_test.cu preprocess_cuda.cu)
    target_include_directories(vb_trt_cuda_reference_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vb_trt_cuda_reference_test PRIVATE CUDA::cudart)
    add_test(NAME vb_trt_cuda_reference_test COMMAND vb_trt_cuda_reference_test)
    add_executable(vb_trt_stage2_test trt_stage2_test.cpp)
    target_include_directories(vb_trt_stage2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vb_trt_stage2_test PRIVATE vb_trt vb_core vb_algo ${TRT_LIBRARY} CUDA::cudart PkgConfig::GST_TRT)
    # Requires a device-specific engine path; invoke explicitly in device validation.
endif()
