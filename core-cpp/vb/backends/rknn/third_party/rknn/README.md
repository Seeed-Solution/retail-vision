# rknn_api.h — fetched, not committed

`rknn_api.h` is a **proprietary Rockchip header** (its own text forbids
reproduction; the rknn-toolkit2 repository `LICENSE` is a Rockchip copyright
statement, not an open-source license — full evidence in
`../THIRD_PARTY.md`). It is therefore not tracked in this repository, and the
build fetches it here instead:

```
curl -fsSL -o rknn_api.h \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
echo "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a  rknn_api.h" | sha256sum -c -
```

`docker/base-rk.Dockerfile` does exactly this in its `build` stage, so a
container build needs no manual step; a local CMake build does.

The header is a compile-time input only. The library it declares
(`librknnrt.so`) is not shipped either: on the boards it is a host mount.
