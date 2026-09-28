# Hailo (Pi 5 + Hailo-8) backend (spec BASE-1 §8 M2.3).
#
# OBJECT library, so the self-registration object in register.cpp is always
# linked into vb-runtime even though nothing references its symbols (same
# pattern as backends/rknn).
#
# Two build modes:
#   VB_HAILO_STUB=ON  (default OFF)  batched_hailo_runner_stub.cpp replaces the
#                     HailoRT session with a fake runner; no HailoRT, no
#                     GStreamer required. This is the offline ctest path.
#   VB_HAILO_STUB=OFF link the real HailoRT: headers and libhailort.so come
#                     from the host (find_package(HailoRT)), never from this
#                     repository and never shipped in an image (spec §M2.3
#                     guardrail; §M2 "镜像与锁定" rule 3). vb-runtime records
#                     libhailort as DT_NEEDED and resolves it at run time.
#
# hailo_source.cpp (GStreamer frame source) needs VB_WITH_GST=ON in both
# modes; without it the backend builds and answers caps/context, and only
# create_source() reports the gap.
option(VB_HAILO_STUB "Build the hailo backend against a fake runner (no HailoRT)" OFF)

set(VB_HAILO_SOURCES
    batch_policy.cpp
    frame_batcher.cpp
    hailo_backend.cpp
    model_sha256.cpp
    register.cpp)

if(VB_HAILO_STUB)
    list(APPEND VB_HAILO_SOURCES batched_hailo_runner_stub.cpp)
    add_library(vb_hailo OBJECT ${VB_HAILO_SOURCES})
    target_compile_definitions(vb_hailo PUBLIC VB_HAILO_STUB=1)
    message(STATUS "vb: hailo backend enabled (STUB, no HailoRT)")
else()
    list(APPEND VB_HAILO_SOURCES batched_hailo_runner.cpp)
    find_package(HailoRT REQUIRED)
    add_library(vb_hailo OBJECT ${VB_HAILO_SOURCES})
    target_link_libraries(vb_hailo PRIVATE HailoRT::libhailort)
    message(STATUS "vb: hailo backend enabled (HailoRT)")
endif()

target_include_directories(vb_hailo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # sources/gst_source.h (url policy constants) is a vb_core private header.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(vb_hailo PRIVATE
    $<TARGET_PROPERTY:vb_core,INTERFACE_COMPILE_DEFINITIONS>)
set_target_properties(vb_hailo PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(VB_WITH_GST)
    target_sources(vb_hailo PRIVATE hailo_source.cpp)
    target_compile_definitions(vb_hailo PRIVATE VB_WITH_GST=1)
    # GStreamer is PRIVATE: vb_core already links it for the generic source,
    # but its usage requirements do not propagate to backend OBJECT libs.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST_HAILO REQUIRED IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    target_link_libraries(vb_hailo PRIVATE PkgConfig::GST_HAILO)
endif()

target_link_libraries(vb-runtime PRIVATE vb_hailo)

if(VB_TESTS AND VB_HAILO_STUB)
    add_executable(test_hailo_backend tests/test_hailo_backend.cpp)
    # vb_hailo carries the self-registration object and the policy/batcher/
    # stub-runner sources the test drives.
    target_link_libraries(test_hailo_backend PRIVATE vb_core vb_hailo)
    target_include_directories(test_hailo_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests
        $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
    add_test(NAME hailo_backend COMMAND test_hailo_backend)
endif()
