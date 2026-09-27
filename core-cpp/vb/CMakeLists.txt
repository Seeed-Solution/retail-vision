cmake_minimum_required(VERSION 3.16)
project(vb C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VB_TESTS "Build vb tests and vb_selftest" ON)
option(VB_WITH_GST "Build the generic GStreamer frame source" OFF)
option(VB_WITH_JPEG "Snapshot JPEG encoding via libjpeg-turbo" OFF)
option(VB_WITH_TLS "Report OpenSSL availability (MQTT TLS lands with M1.19)" ON)
set(VB_BACKENDS "cpu" CACHE STRING "Semicolon-separated backend list")
set(ORT_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime install root (cpu backend)")

find_package(Threads REQUIRED)

# §6.13.2 (M1.24): vb_algo — installable compute-only static library
# (letterbox, tracker, analyzers incl. plugin loader, wire, post, geom2d).
# No frame sources, no thread pool, no backends, no ORT/GStreamer.
add_library(vb_algo STATIC
    src/letterbox.cpp
    src/json.cpp
    src/wire.cpp
    src/tracker.cpp
    src/analyzers/line_cross.cpp
    src/analyzers/zone.cpp
    src/analyzers/plugin_loader.cpp
    src/geom2d.cpp
    src/post/nms.cpp
    src/post/yolox_decode.cpp
    src/post/yolov8_decode.cpp
    src/post/classify.cpp
    src/post/decoder.cpp
    src/analyzers/motion_util.cpp
    src/analyzers/dwell.cpp
    src/analyzers/speed.cpp
    src/analyzers/direction.cpp
    src/analyzers/count_threshold.cpp
    src/analyzers/pose_angle.cpp
)
add_library(vb::algo ALIAS vb_algo)
set_target_properties(vb_algo PROPERTIES EXPORT_NAME algo)
target_compile_features(vb_algo PUBLIC cxx_std_17)
target_include_directories(vb_algo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/vb/third_party>)
target_include_directories(vb_algo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vb_algo PUBLIC ${CMAKE_DL_LIBS})

# vb_runtime library: frame sources, pool/runtime scheduling, output
# adapters; links the compute-only vb_algo instead of recompiling sources.
add_library(vb_core STATIC
    src/latest_frame.cpp
    src/writer.cpp
    src/stats.cpp
    src/pool.cpp
    src/runtime.cpp
    src/backend_registry.cpp
    src/sources/synthetic.cpp
    src/dev_tensor.cpp
    src/snapshot.cpp
    src/out/standalone.cpp
    src/out/event_json.cpp
    src/out/jsonl_out.h
    src/out/mqtt_lite.cpp
)
target_include_directories(vb_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vb_core PUBLIC vb_algo ${CMAKE_DL_LIBS} Threads::Threads)

# §6.10.3: TLS is a compile option (default ON, OpenSSL >= 1.1.1); it backs
# the MQTT TLS transport in src/out/mqtt_lite.cpp and the tls:on|off marker
# in `vb-runtime --version`.
if(VB_WITH_TLS)
    find_package(OpenSSL 1.1.1)
    if(OpenSSL_FOUND)
        target_compile_definitions(vb_core PRIVATE VB_WITH_TLS=1)
        target_link_libraries(vb_core PRIVATE OpenSSL::SSL OpenSSL::Crypto)
        message(STATUS "vb: TLS enabled (OpenSSL ${OPENSSL_VERSION})")
    else()
        message(STATUS "vb: OpenSSL not found, TLS disabled (tls:off)")
    endif()
else()
    message(STATUS "vb: TLS disabled (VB_WITH_TLS=OFF)")
endif()

if(VB_WITH_GST)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    target_sources(vb_core PRIVATE src/sources/gst_source.cpp)
    target_compile_definitions(vb_core PRIVATE VB_WITH_GST=1)
    target_link_libraries(vb_core PRIVATE PkgConfig::GST)
    message(STATUS "vb: GStreamer frame source enabled")
endif()

if(VB_WITH_JPEG)
    find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h REQUIRED)
    find_library(TURBOJPEG_LIB turbojpeg REQUIRED)
    target_compile_definitions(vb_core PRIVATE VB_HAVE_JPEG=1)
    target_include_directories(vb_core PRIVATE ${TURBOJPEG_INCLUDE_DIR})
    # Private link so the definition above reaches snapshot.cpp.
    target_sources(vb_core PRIVATE src/snapshot.cpp)
    set_source_files_properties(src/snapshot.cpp PROPERTIES SKIP_LINT ON)
    get_target_property(_vb_core_srcs vb_core SOURCES)
    list(REMOVE_DUPLICATES _vb_core_srcs)
    set_target_properties(vb_core PROPERTIES SOURCES "${_vb_core_srcs}")
    target_link_libraries(vb_core PRIVATE ${TURBOJPEG_LIB})
    message(STATUS "vb: snapshot JPEG enabled (turbojpeg: ${TURBOJPEG_LIB})")
else()
    message(STATUS "vb: snapshot JPEG disabled (VB_WITH_JPEG=OFF)")
endif()

add_executable(vb-runtime src/main.cpp)
target_link_libraries(vb-runtime PRIVATE vb_core)
if(VB_WITH_TLS AND OpenSSL_FOUND)
    target_compile_definitions(vb-runtime PRIVATE VB_WITH_TLS=1)
endif()

# ---- vb_algo install/export (spec BASE-1 §6.13.2, M1.24) ----
include(CMakePackageConfigHelpers)
install(TARGETS vb_algo EXPORT vbTargets ARCHIVE DESTINATION lib)
install(DIRECTORY include/vb DESTINATION include)
install(FILES third_party/nlohmann/json.hpp DESTINATION include/vb/third_party/nlohmann)
install(EXPORT vbTargets NAMESPACE vb:: DESTINATION lib/cmake/vb)
configure_package_config_file(cmake/vbConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vbConfig.cmake INSTALL_DESTINATION lib/cmake/vb)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vbConfig.cmake DESTINATION lib/cmake/vb)


# Tests must be enabled before the backend subdirectories add theirs.
if(VB_TESTS)
    enable_testing()
endif()

# Platform inference adapters live in backends/<p>/; core code never branches
# on backend names (spec BASE-1 §5.2.3). Backend sources not present yet are
# skipped so the skeleton builds standalone (M1.3).
foreach(_backend IN LISTS VB_BACKENDS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backends/${_backend}")
        add_subdirectory("backends/${_backend}")
        message(STATUS "vb: backend ${_backend} enabled")
    else()
        message(STATUS "vb: backend ${_backend} sources absent, skipping")
    endif()
endforeach()

if(VB_TESTS)
    enable_testing()

    add_executable(vb_selftest tests/selftest.cpp)
    target_link_libraries(vb_selftest PRIVATE vb_core)

    add_executable(test_pool tests/test_pool.cpp)
    target_link_libraries(test_pool PRIVATE vb_core)
    add_test(NAME pool COMMAND test_pool)

    add_executable(synthetic_backend tests/synthetic_backend.cpp)
    target_link_libraries(synthetic_backend PRIVATE vb_core)
    add_test(NAME synthetic_backend COMMAND synthetic_backend)

    add_executable(test_control tests/test_control.cpp)
    target_link_libraries(test_control PRIVATE vb_core)
    add_test(NAME control COMMAND test_control)

    add_executable(test_wire tests/test_wire.cpp)
    target_link_libraries(test_wire PRIVATE vb_core)
    add_test(NAME wire COMMAND test_wire ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb)

    add_executable(test_dev_tensor tests/test_dev_tensor.cpp)
    target_link_libraries(test_dev_tensor PRIVATE vb_core)
    add_test(NAME dev_tensor COMMAND test_dev_tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb
        $<TARGET_FILE:vb-runtime>)

    add_executable(vb_wire_gen tools/vb_wire_gen.cpp)
    target_link_libraries(vb_wire_gen PRIVATE vb_core)

    add_executable(test_tracker tests/test_tracker.cpp)
    target_link_libraries(test_tracker PRIVATE vb_core)
    add_test(NAME tracker COMMAND test_tracker
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core-py/retail_core/tests/fixtures/tracker_sequence.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb/tracker_generic.json)

    add_executable(test_analyzers tests/test_analyzers.cpp)
    target_link_libraries(test_analyzers PRIVATE vb_core)

    add_executable(test_motion tests/test_motion.cpp)
    target_link_libraries(test_motion PRIVATE vb_core)
    add_test(NAME motion COMMAND test_motion
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb/analyzers)

    add_executable(test_count_pose tests/test_count_pose.cpp)
    target_link_libraries(test_count_pose PRIVATE vb_core)
    add_test(NAME count_pose COMMAND test_count_pose
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb/analyzers)

    add_executable(test_out tests/test_out.cpp)
    target_link_libraries(test_out PRIVATE vb_core)
    add_test(NAME out COMMAND test_out)

    add_executable(test_runtime tests/test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE vb_core)
    add_test(NAME runtime COMMAND test_runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb)

    add_executable(test_sources tests/test_sources.cpp)
    target_link_libraries(test_sources PRIVATE vb_core)
    add_test(NAME sources COMMAND test_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb)

    add_executable(test_decoder tests/test_decoder.cpp)
    target_link_libraries(test_decoder PRIVATE vb_core)
    add_test(NAME decoder COMMAND test_decoder
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb)

    add_library(count_plugin MODULE tests/plugins/count_plugin.c)
    set_target_properties(count_plugin PROPERTIES PREFIX "" SUFFIX ".so")
    target_include_directories(count_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_library(count_plugin_bad MODULE tests/plugins/count_plugin.c)
    set_target_properties(count_plugin_bad PROPERTIES PREFIX "" SUFFIX ".so")
    target_include_directories(count_plugin_bad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(count_plugin_bad PRIVATE VB_TEST_BAD_ABI=1)

    add_test(NAME analyzers COMMAND test_analyzers
        ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts/fixtures/vb/analyzers
        $<TARGET_FILE:count_plugin>
        $<TARGET_FILE:count_plugin_bad>)
endif()
