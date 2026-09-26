# Third-party notices — core-cpp/vb

## ONNX Runtime (CPU backend, M1.9)

- Component: ONNX Runtime, version 1.20.1 (used via the C API, `<onnxruntime_c_api.h>`)
- License: MIT (https://github.com/microsoft/onnxruntime/blob/v1.20.1/LICENSE)
- Downloaded at build time from the official GitHub release; the tarball is
  **not** committed to this repository. Build with
  `-DORT_ROOT=<extracted tarball root>` (CI keeps it under /tmp):
  - URL: https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz
  - sha256 (osx-arm64 tarball, recorded 2026-09-26):
    `b678fc3c2354c771fea4fba420edeccfba205140088334df801e7fc40e83a57a`
  - Other platforms: substitute the matching `onnxruntime-<os>-<arch>-1.20.1.tgz`
    from the same release page (M1.14 pins the full list for the CI image).

## nlohmann/json

- Vendored single header under `third_party/nlohmann/`; see
  `third_party/nlohmann/THIRD_PARTY.md` (M1.3).
