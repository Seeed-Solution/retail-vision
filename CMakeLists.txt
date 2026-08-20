cmake_minimum_required(VERSION 3.18)
project(retail_vision LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# retail-vision-core: tracking, dwell state machine and payload construction.
# Carried over from solutions/retail-vision/main with two one-line include
# changes and no logic edits, so every backend that links it produces the same
# track ids, the same dwell transitions and the same JSON. Behaviour is shared
# because the object code is shared, not because two implementations were
# tested against each other.
# ---------------------------------------------------------------------------
add_library(retail-vision-core STATIC
  core-cpp/person_tracker.cpp
  core-cpp/zone_metrics.cpp
  core-cpp/mqtt_payload.cpp)
target_include_directories(retail-vision-core PUBLIC core-cpp)
target_compile_options(retail-vision-core PRIVATE -Wall -Wextra -O2)

# ---------------------------------------------------------------------------
# Hailo backend: the only part written for this board. GStreamer feeds hailonet,
# hailonet runs the HEF on the NPU, and the decoder turns its output tensor into
# the same std::vector<DetectionBox> the SG2002 detector produces.
# ---------------------------------------------------------------------------
option(ENABLE_MQTT "Build the libmosquitto publisher" ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-video-1.0)
find_package(HailoRT REQUIRED)
find_library(HAILO_GST_LIBRARY gsthailo
             PATHS /usr/lib/aarch64-linux-gnu/gstreamer-1.0 REQUIRED)

add_executable(retail-vision-hailo
  backends/hailo/main.cpp
  backends/hailo/detector.cpp)
target_include_directories(retail-vision-hailo PRIVATE backends/hailo)
target_link_libraries(retail-vision-hailo PRIVATE
  retail-vision-core PkgConfig::GST HailoRT::libhailort ${HAILO_GST_LIBRARY} pthread)

if(ENABLE_MQTT)
  find_library(MOSQUITTO_LIBRARY mosquitto REQUIRED)
  target_compile_definitions(retail-vision-hailo PRIVATE HAVE_MOSQUITTO=1)
  target_link_libraries(retail-vision-hailo PRIVATE ${MOSQUITTO_LIBRARY})
endif()

target_compile_options(retail-vision-hailo PRIVATE -Wall -Wextra -O2)
