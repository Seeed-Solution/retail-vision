# RK3576 / RK3588 backend (spec BASE-1 §8 M2.1).
#
# OBJECT library, so the self-registration object in register.cpp is always
# linked into vb-runtime even though nothing references its symbols.
#
# Two vendor API sets are needed:
#   headers   third_party/rga/*.h        vendored, Apache-2.0
#             third_party/rknn/rknn_api.h fetched at build time (proprietary;
#                                         see third_party/THIRD_PARTY.md)
#   libraries librknnrt.so, librga.so    fetched at build time into
#                                         RKNN_LIB_DIR / RGA_LIB_DIR and NOT
#                                         shipped: the boards provide both as
#                                         host mounts (spec §M2 "镜像与锁定"
#                                         rule 3), so vb-runtime records them
#                                         as DT_NEEDED and resolves them from
#                                         the runtime image's mount points.
# Linking against the real libraries (not a stub) is what makes a mistyped
# vendor symbol a build error instead of a runtime one.
set(RKNN_LIB_DIR "/opt/rk" CACHE PATH "directory holding librknnrt.so")
set(RGA_LIB_DIR "/opt/rk" CACHE PATH "directory holding librga.so")
set(RKNN_API_INCLUDE_DIR "" CACHE PATH
    "directory containing the proprietary rknn_api.h header")

if(RKNN_API_INCLUDE_DIR)
    if(NOT EXISTS "${RKNN_API_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR
            "vb: RKNN_API_INCLUDE_DIR must contain rknn_api.h: "
            "${RKNN_API_INCLUDE_DIR}")
    endif()
    set(_RKNN_API_INCLUDE_DIR "${RKNN_API_INCLUDE_DIR}")
elseif(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/rknn_api.h")
    message(FATAL_ERROR
        "vb: rknn backend needs third_party/rknn/rknn_api.h. It is a "
        "proprietary Rockchip header and is deliberately not committed; fetch "
        "it with the URL and sha256 in "
        "backends/rknn/third_party/THIRD_PARTY.md.")
else()
    set(_RKNN_API_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn")
endif()

find_library(RKNN_LIBRARY NAMES rknnrt librknnrt.so
             HINTS "${RKNN_LIB_DIR}" REQUIRED)
find_library(RGA_LIBRARY NAMES rga librga.so HINTS "${RGA_LIB_DIR}" REQUIRED)

add_library(vb_rknn OBJECT
    hybrid_rga_rknn.cpp
    rknn_stage2.cpp
    rknn_backend.cpp
    mpp_source.cpp
    model_sha256.cpp
    register.cpp)
target_include_directories(vb_rknn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${_RKNN_API_INCLUDE_DIR})
# Only vb_core's usage requirements, not the library itself (same reason as the
# cpu backend: an OBJECT library forwards its link deps to every consumer).
target_include_directories(vb_rknn PRIVATE
    $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(vb_rknn PRIVATE
    $<TARGET_PROPERTY:vb_core,INTERFACE_COMPILE_DEFINITIONS>)
if(NOT VB_WITH_GST)
    message(FATAL_ERROR "vb: the rknn backend needs -DVB_WITH_GST=ON (MPP decode)")
endif()
# GStreamer is PRIVATE here; vb_core already links it PUBLIC-ly for the generic
# source, and the allocators module is only used by mpp_source.cpp.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_ALLOC REQUIRED IMPORTED_TARGET gstreamer-allocators-1.0)
target_link_libraries(vb_rknn PRIVATE PkgConfig::GST_ALLOC)
target_link_libraries(vb_rknn PRIVATE ${RKNN_LIBRARY} ${RGA_LIBRARY})
set_target_properties(vb_rknn PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(vb-runtime PRIVATE vb_rknn)
message(STATUS "vb: rknn backend enabled (librknnrt: ${RKNN_LIBRARY}, "
               "librga: ${RGA_LIBRARY})")
