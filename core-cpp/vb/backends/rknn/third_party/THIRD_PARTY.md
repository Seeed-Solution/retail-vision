# Third-party notices — core-cpp/vb/backends/rknn

Two vendor API header sets are needed to build the RK3576/RK3588 backend. Only
one of them may be redistributed from this repository.

## Rockchip librga headers (`rga/`) — vendored, Apache-2.0

- Component: librga `include/` (18 headers), upstream `airockchip/librga`
- Pinned revision: tag **`v1.10.0`**, commit
  `fb3357d09008222bc5e27bdaadf74a0c5ea4c86e` (verified 2026-09-28)
- License: **Apache License 2.0** — verified 2026-09-28 from two independent
  places in the upstream tree:
  - `COPYING` (repository root) is the unmodified Apache License 2.0 text;
    it is vendored here verbatim as `rga/COPYING`.
  - every vendored header carries the per-file notice
    `Copyright (C) <year> Rockchip Electronics Co., Ltd.` +
    `Licensed under the Apache License, Version 2.0`.
  - GitHub's license API independently reports `spdx_id: Apache-2.0` for the
    repository (`/repos/airockchip/librga`).
- **Redistribution: permitted.** Apache-2.0 allows redistribution of the
  headers with the copyright and license notices retained (§4). The notices
  are in the files themselves and `rga/COPYING` is included.
- Version compatibility: the headers declare `rga_api version 1.10.0_[2]`
  (`rga/im2d_version.h`). The runtime shipped on the measured boards is
  `rga_api version 1.10.1_[4]` (`librga.so.2.1.0`). Same minor, library newer
  than the header — the direction librga's `imcheckHeader()` accepts.
- Files (sha256, all fetched from the pinned commit):

```
5c98dd41033435de2f25e54324ded7f134f3031e94cd0eef60c71627ce225a0d  GrallocOps.h
b71656a3919c9e327de9b13f818dfda11d47eb781535edbcbc4b839289df3cb8  RgaApi.h
b40e7b67aaf96d81dc26ff1a9bfdd3bf1b11a7569286303e7c369c8fd74a6d75  RgaMutex.h
27f65fcf1472baf08683707bc7a75dbbba85d37cc347b6fd24535f5e194f8325  RgaSingleton.h
4537e1e7840e192f75b8678599b6217421f56fea58199af602ab137241f4536d  RgaUtils.h
cc386291fb0585006917094f153923a23963d9c810dc8100bc96d9a91b0436bf  RockchipRga.h
7db37dfc3fedc4573cf0d223f87ea319a66d7af90670b8a259248bec52b3eb15  drmrga.h
14f9ed0da92fc0867fa50f1b397cf33640bd58f75d450d5e1ff2fe103875d3b9  im2d_buffer.h
788a91fc2e67012df8252f6e02c75292a44317131ccd6672f8db1ac851effd19  im2d_common.h
5ff151ccccc35eba17019096bf60ac077dd353ff3c8bda0cb90540d0c98c1dc6  im2d_expand.h
820484df831f12f9d44f673bee7f6f180470d828e9233527c9d70ef59d2fe3f7  im2d_mpi.h
38742650853c5ebdf4a87a8c6453806bc82130816f02b4102e91434773ad258e  im2d_single.h
176feab418ab9abef15917bbb0d678acd680089e0515a86fc5a4417bc9627e38  im2d_task.h
54260b7650c4af9281f4f201e63d6c663372365aecdab899f54edcd6e7bf1e4f  im2d_type.h
82785c4b68a86d4f5abc91bc17397ab7e24f2163a65f8e6285efbab938441351  im2d_version.h
5ee06891775451c56e6264f5d53db0a4e948e1830287f3a5db4f75c30ac0ff11  im2d.h
d4c3f3aa52ed517d3185768ee1c1b9fe0b6791e4047d8f695a62976c87730a1b  rga.h
878d975e8596fa8dbea20dd18b5d1a817b1a258aef3ca924dc42cc174fc6834d  im2d.hpp
```

## Rockchip RKNN runtime header (`rknn/`) — NOT committed, fetched at build time

- Component: `rknn_api.h`, upstream `airockchip/rknn-toolkit2`
- Pinned revision: tag **`v2.3.2`**, path
  `rknpu2/runtime/Linux/librknn_api/include/rknn_api.h`
- URL: `https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h`
- sha256 (verified 2026-09-28): `c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a`
- **License: proprietary. It is deliberately not redistributed from this
  repository.** Evidence, both from upstream:
  - the repository root `LICENSE` (v2.3.2) is a Rockchip *Copyright
    Statement*, not an open-source license: "Copyright(C) 2024 Rockchip
    Electronics Co., Ltd. All rights reserved." with an AS-IS disclaimer and
    no grant of redistribution rights. GitHub's license API reports
    `spdx_id: NOASSERTION` / `key: other` for the repository.
  - `rknn_api.h` itself opens with: "The material in this file is confidential
    and contains trade secrets of Rockchip Corporation. This is proprietary
    information owned by Rockchip Corporation. No part of this work may be
    disclosed, reproduced, copied, transmitted, or used in any way for any
    purpose, without the express prior written permission of Rockchip
    Corporation."
- Consequence: the header is **fetched and sha256-verified at build time** into
  `third_party/rknn/` (gitignored), the same pattern this repository already
  uses for ONNX Runtime in `docker/base-cpu.Dockerfile` and
  `third_party/THIRD_PARTY.md`. It is not a tracked file, and it is not copied
  into any image: the runtime `.so` stays a host mount, matching the
  fall-detection RK packaging rule ("proprietary headers and vendor libraries
  are intentionally excluded").
- For a local (non-Docker) build, point CMake at an extracted copy with
  `-DRKNN_API_INCLUDE_DIR=<dir containing rknn_api.h>`.
