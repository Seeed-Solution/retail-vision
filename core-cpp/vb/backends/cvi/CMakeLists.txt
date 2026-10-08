# CVITEK CVI_NN/VI/VPSS backend. SDK paths are explicit: no host defaults.
set(CVI_TPU_ROOT "" CACHE PATH "CVI TPU SDK root containing include/cviruntime.h and lib")
set(CVI_MPI_ROOT "" CACHE PATH "CVI MPI root containing cvi_* headers and libraries")
set(SSCMA_VIDEO_ROOT "" CACHE PATH "Existing SSCMA sophgo/video component root")
set(SSCMA_OSDRV_ROOT "" CACHE PATH "Existing SG2002 SDK osdrv root")
if(NOT EXISTS "${SSCMA_OSDRV_ROOT}/interdrv/include/common/uapi")
  message(FATAL_ERROR "vb cvi: set SSCMA_OSDRV_ROOT to matching SG2002 osdrv root")
endif()
if(NOT EXISTS "${SSCMA_VIDEO_ROOT}/video.h")
  message(FATAL_ERROR "vb cvi: set SSCMA_VIDEO_ROOT to existing SSCMA sophgo/video component")
endif()
set(SSCMA_VIDEO_SOURCES "${SSCMA_VIDEO_ROOT}/video.c")
file(GLOB SSCMA_VIDEO_SRC_FILES "${SSCMA_VIDEO_ROOT}/src/*.c" "${SSCMA_VIDEO_ROOT}/../common/*.c")
list(APPEND SSCMA_VIDEO_SOURCES ${SSCMA_VIDEO_SRC_FILES})
if(NOT EXISTS "${CVI_TPU_ROOT}/include/cviruntime.h")
  message(FATAL_ERROR "vb cvi: set CVI_TPU_ROOT to the CVI TPU SDK root")
endif()
if(NOT EXISTS "${CVI_MPI_ROOT}/include/cvi_vi.h")
  message(FATAL_ERROR "vb cvi: set CVI_MPI_ROOT to the CVI MPI root")
endif()
find_library(CVI_RUNTIME_LIBRARY NAMES cviruntime HINTS "${CVI_TPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_VI_LIBRARY NAMES vi HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_VPSS_LIBRARY NAMES vpss HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_SYS_LIBRARY NAMES sys HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_VO_LIBRARY NAMES vo HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_ISP_LIBRARY NAMES isp HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
set(CVI_VENDOR_LIBS ae af awb cvi_bin cvi_bin_isp gdc isp_algo venc raw_dump vdec rgn cvi_ispd2 sns_ov5647 sns_gc2053)
foreach(_lib IN LISTS CVI_VENDOR_LIBS)
  find_library(CVI_${_lib}_LIBRARY NAMES ${_lib} HINTS "${CVI_MPI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  list(APPEND CVI_VENDOR_LINK_LIBS ${CVI_${_lib}_LIBRARY})
endforeach()
find_library(CVI_INI_LIBRARY NAMES ini HINTS "${CVI_MPI_ROOT}/lib/3rd" NO_DEFAULT_PATH REQUIRED)
find_library(CVI_JSONC_LIBRARY NAMES json-c HINTS "${CVI_TPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
list(APPEND CVI_VENDOR_LINK_LIBS ${CVI_INI_LIBRARY} ${CVI_JSONC_LIBRARY})
add_library(vb_cvi OBJECT cvi_backend.cpp cvi_source.cpp cvi_stage2.cpp cvi_resize.cpp cvi_sha256.cpp register.cpp ${SSCMA_VIDEO_SOURCES})
target_include_directories(vb_cvi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CVI_TPU_ROOT}/include ${CVI_MPI_ROOT}/include ${CVI_MPI_ROOT}/include/linux
  ${SSCMA_VIDEO_ROOT} ${SSCMA_VIDEO_ROOT}/include ${SSCMA_VIDEO_ROOT}/../common
  ${CVI_MPI_ROOT}/modules/isp/include/cv181x ${CVI_MPI_ROOT}/modules/isp/include
  ${CVI_MPI_ROOT}/modules/isp/common/raw_dump/inc
  ${SSCMA_OSDRV_ROOT}/interdrv/include/common/uapi
  ${SSCMA_OSDRV_ROOT}/interdrv/include/common/uapi/linux
  ${SSCMA_OSDRV_ROOT}/interdrv/include/chip/cv181x/uapi
  $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(vb_cvi PRIVATE $<TARGET_PROPERTY:vb_core,INTERFACE_COMPILE_DEFINITIONS> VB_CVI_SSCMA_VIDEO)
target_link_libraries(vb_cvi PRIVATE ${CVI_RUNTIME_LIBRARY} ${CVI_VI_LIBRARY} ${CVI_VPSS_LIBRARY} ${CVI_SYS_LIBRARY} ${CVI_VO_LIBRARY} ${CVI_ISP_LIBRARY} ${CVI_VENDOR_LINK_LIBS} atomic)
set_target_properties(vb_cvi PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vb-runtime PRIVATE vb_cvi ${CVI_RUNTIME_LIBRARY} ${CVI_VI_LIBRARY} ${CVI_VPSS_LIBRARY} ${CVI_SYS_LIBRARY} ${CVI_VO_LIBRARY} ${CVI_ISP_LIBRARY} ${CVI_VENDOR_LINK_LIBS} atomic)
