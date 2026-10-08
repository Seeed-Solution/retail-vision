# CPU backend (spec BASE-1 §8 M1.9): ONNX Runtime C API.
# ORT_ROOT must point at an extracted onnxruntime-<plat>-1.20.1 tarball root
# (include/ + lib/). OBJECT library so the self-registration object in
# register.cpp is always linked into vb-runtime even though nothing
# references its symbols.
find_path(ORT_INCLUDE_DIR onnxruntime_c_api.h HINTS "${ORT_ROOT}/include" REQUIRED)
find_library(ORT_LIBRARY onnxruntime HINTS "${ORT_ROOT}/lib" REQUIRED)

add_library(vb_cpu OBJECT cpu_backend.cpp cpu_stage2.cpp register.cpp sha256.cpp)
target_include_directories(vb_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ORT_INCLUDE_DIR})
# Only vb_core's usage requirements, not the library itself: an OBJECT
# library forwards its link deps to every consumer, which made vb_core
# appear twice on the vb-runtime / test link lines (M1.14 review).
target_include_directories(vb_cpu PRIVATE $<TARGET_PROPERTY:vb_core,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(vb_cpu PRIVATE $<TARGET_PROPERTY:vb_core,INTERFACE_COMPILE_DEFINITIONS>)
target_link_libraries(vb_cpu PRIVATE ${ORT_LIBRARY})
if(VB_WITH_GST)
    target_compile_definitions(vb_cpu PRIVATE VB_WITH_GST=1)
endif()
set_target_properties(vb_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

message(STATUS "vb: cpu backend enabled (ort: ${ORT_LIBRARY})")

if(VB_TESTS)
    add_executable(test_cpu_stage2 ../../tests/test_cpu_stage2.cpp)
    target_include_directories(test_cpu_stage2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ORT_INCLUDE_DIR})
    target_link_libraries(test_cpu_stage2 PRIVATE vb_core vb_cpu)
    find_program(VB_STAGE2_UV uv REQUIRED)
    set(VB_STAGE2_FIXTURES "${CMAKE_BINARY_DIR}/stage2-models")
    add_custom_command(OUTPUT "${VB_STAGE2_FIXTURES}/pixels2.onnx"
        COMMAND "${VB_STAGE2_UV}" run --no-sync --project "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
            python "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/gen_stage2_test_models.py" "${VB_STAGE2_FIXTURES}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/gen_stage2_test_models.py" VERBATIM)
    add_custom_target(vb_stage2_fixtures DEPENDS "${VB_STAGE2_FIXTURES}/pixels2.onnx")
    add_dependencies(test_cpu_stage2 vb_stage2_fixtures)
    add_test(NAME cpu_stage2 COMMAND test_cpu_stage2 "${VB_STAGE2_FIXTURES}")

    add_executable(test_cpu_backend ../../tests/test_cpu_backend.cpp)
    target_include_directories(test_cpu_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ORT_INCLUDE_DIR})
    target_link_libraries(test_cpu_backend PRIVATE vb_core vb_cpu)
    if(VB_WITH_GST)
        target_compile_definitions(test_cpu_backend PRIVATE VB_WITH_GST=1)
    endif()

    # M1.9: test_cpu_backend Part B needs a constant-output ONNX graph built
    # from the yolox_out_case1 fixture. Generate it at build time instead of
    # requiring a manual `uv run python ...` step, so `ctest -R cpu_backend`
    # is self-contained on a checkout that has `uv` on PATH.
    set(VB_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
    set(VB_GEN_CONST_ONNX "${VB_REPO_ROOT}/core-py/vision_base/tests/gen_const_onnx.py")
    set(VB_CONST_ONNX_FIXTURE "${VB_REPO_ROOT}/contracts/fixtures/vb/yolox_out_case1")
    set(VB_CONST_ONNX "${CMAKE_BINARY_DIR}/const.onnx")
    find_program(VB_UV_EXECUTABLE uv)
    if(VB_UV_EXECUTABLE)
        add_custom_command(
            OUTPUT "${VB_CONST_ONNX}"
            COMMAND "${VB_UV_EXECUTABLE}" run --project "${VB_REPO_ROOT}" python
                    "${VB_GEN_CONST_ONNX}" "${VB_CONST_ONNX}"
            DEPENDS "${VB_GEN_CONST_ONNX}" "${VB_CONST_ONNX_FIXTURE}.json" "${VB_CONST_ONNX_FIXTURE}.bin"
            WORKING_DIRECTORY "${VB_REPO_ROOT}"
            COMMENT "vb: generating const.onnx for test_cpu_backend (uv run gen_const_onnx.py)"
            VERBATIM)
        add_custom_target(vb_cpu_const_onnx DEPENDS "${VB_CONST_ONNX}")
        add_dependencies(test_cpu_backend vb_cpu_const_onnx)
    else()
        message(WARNING "vb: 'uv' not found on PATH; const.onnx cannot be generated "
                        "at build time. test_cpu_backend will report SKIP (ctest "
                        "SKIP_RETURN_CODE 77) unless ${VB_CONST_ONNX} is generated "
                        "manually (uv run python ${VB_GEN_CONST_ONNX} ${VB_CONST_ONNX})")
    endif()

    add_test(NAME cpu_backend COMMAND test_cpu_backend
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../contracts/fixtures/vb
        ${VB_CONST_ONNX})
    # Belt-and-braces: even with the custom command above, a stale/removed
    # const.onnx (or a non-uv build tree) must SKIP rather than FAIL --
    # test_cpu_backend returns 77 when the file is missing.
    set_tests_properties(cpu_backend PROPERTIES SKIP_RETURN_CODE 77)

    # Production preprocessor oracle: an identity graph exposes the actual
    # NCHW tensor produced by CpuBackend, including padding, stride and colour
    # order.  The fixture is generated with the repository's uv environment.
    set(VB_CPU_PREPROCESS_GEN
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/gen_cpu_preprocess_model.py")
    set(VB_CPU_PREPROCESS_4 "${CMAKE_BINARY_DIR}/cpu-preprocess-4.onnx")
    set(VB_CPU_PREPROCESS_5 "${CMAKE_BINARY_DIR}/cpu-preprocess-5.onnx")
    if(VB_UV_EXECUTABLE)
        add_custom_command(OUTPUT "${VB_CPU_PREPROCESS_4}"
            COMMAND "${VB_UV_EXECUTABLE}" run --no-sync --project "${VB_REPO_ROOT}"
                    python "${VB_CPU_PREPROCESS_GEN}" "${VB_CPU_PREPROCESS_4}" 4
            DEPENDS "${VB_CPU_PREPROCESS_GEN}" VERBATIM)
        add_custom_command(OUTPUT "${VB_CPU_PREPROCESS_5}"
            COMMAND "${VB_UV_EXECUTABLE}" run --no-sync --project "${VB_REPO_ROOT}"
                    python "${VB_CPU_PREPROCESS_GEN}" "${VB_CPU_PREPROCESS_5}" 5
            DEPENDS "${VB_CPU_PREPROCESS_GEN}" VERBATIM)
        add_custom_target(vb_cpu_preprocess_fixtures
            DEPENDS "${VB_CPU_PREPROCESS_4}" "${VB_CPU_PREPROCESS_5}")
        add_executable(test_cpu_preprocess ../../tests/test_cpu_preprocess.cpp)
        target_include_directories(test_cpu_preprocess PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${ORT_INCLUDE_DIR})
        target_link_libraries(test_cpu_preprocess PRIVATE vb_core vb_cpu)
        add_dependencies(test_cpu_preprocess vb_cpu_preprocess_fixtures)
        add_test(NAME cpu_preprocess COMMAND test_cpu_preprocess
            "${VB_CPU_PREPROCESS_4}" "${VB_CPU_PREPROCESS_5}")
    endif()
endif()
