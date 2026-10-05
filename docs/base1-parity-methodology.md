# BASE-1 RK parity 方法论定版（引擎可信结论 + gate 口径修正 + 可复用 B1 recipe）

日期：2026-10-05 · 分支 `feat/vision-base`（文档新增，未 commit）
本文把 BASE-1 M2.1「RK 后端是否可信」的全部实测结论沉淀成方法论文档，供 BASE-1 owner 直接采用。**每一条数字/结论均可追到列出的实测报告原文**；报告间如有矛盾，以 `report-engine-row0-vs-harness.md`（决定性实验）为最终结论，其余如实标注。

涉及报告清单（全部存在）：

- `<parity-run-dir>/report-base1-m21-rk-parity-rerun.md`（M2.1 复跑，相位问题首次暴露）
- `<parity-run-dir>/report-base1-gate2-faithfulness.md`（gate 2 不通过 + 初版根因假设，后已证伪）
- `<parity-run-dir>/report-base1-rknn-reconvert.md`（std 255→1 在设备上无效）
- `$WORK_DIR/offline-ab/RESULT-standalone-harness.md`（独立 harness：布局/反量化无责、fp16 忠实、int8 box 头受损）
- `$WORK_DIR/evidence/quant-int8-fix-report.md`（扩 calib / kl_div / OL2 全部无效）
- `$WORK_DIR/hq-eval/RESULT-head-hybrid-fp16.md`（head/PAN 转 fp16 无效）
- `<parity-run-dir>/report-rknn-b1-input-int8-w8a16.md`（B1 离线达标 + 设备 gate 仍低）
- `<parity-run-dir>/report-engine-row0-vs-harness.md`（**决定性**：引擎行 0 ≡ 独立 harness；gate 天花板成因给出画布差异假设，见 §4）
- 夹具提交：`8d08bf4`（对齐夹具）→ `7a12eb6`（内容相位校验）→ `ed40553`（修 tpad/时间戳回归），对应报告 `report-parity-fixture-phase-check.md`；`report-parity-feed-regression-fix.md` **该文件不存在于 <parity-run-dir> 目录**（仅有 task 同名文件），其内容以提交 `ed40553` 为准。

---

## 1. 一句话结论

**引擎的 RKNN 执行 / 反量化 / 解码路径与独立 harness 逐位等价，M2.1 的 RK 后端可信。**

决定性证据（`report-engine-row0-vs-harness.md`）：同一帧（引擎 `VB_RK_DUMP_CANVAS` dump 画布，sha256 `4a2e80fd…`）上，引擎 parity 行 0 vs RKNN 独立 harness（radxa `rknn_dump`，int8 native 喂入）：**6/6 框逐框 IoU = 1.0000，score 小数后 4 位全同**（0.8873/0.8873/0.6534/0.6161/0.4076/0.3449）——引擎从 NPU 执行、fp16 输出读取（`want_float` 转换）、TensorView、grid 解码（strides 8/16/32）、thr 0.3、class-aware NMS 0.45 整条链路与独立 harness 完全等价。

## 2. 模型侧根因：w8a8 激活量化打坏 box 回归头

- **现象**（`RESULT-standalone-harness.md`）：INT8（w8a8）模型误差集中在 box 回归通道 0–3（tx/ty/tw/th）：通道均值 top5 全是 box 通道（0.149/0.136/0.084/0.074），obj(4)≈0.0006–0.0033、cls≈0.005 几乎无损；stride 8/16/32 均匀分布，非某层/某 stride 特有；解码 meanIoU 仅 **0.70**（min 0.516，小框 y2 偏 ~6px）。box(0-3) 平均误差 **0.11**。
- **排除项**（逐条实测无效，见 `quant-int8-fix-report.md` 与 `RESULT-head-hybrid-fp16.md`）：
  - 扩 calib（24 图 → 223 图）：box 0.1527、IoU 0.7046，未达标；
  - `kl_divergence`：0.1954 / 0.6850，**更差**；
  - OL3→OL2：0.1527 / 0.7046，无差；
  - head 24 conv（A1）、+concat/sigmoid 输入（A2）、+FPN/PAN 全部 conv（A3）转 fp16：0.1373–0.1419 / ~0.70，与 int8 基线同量级甚至略差；
  - `quantized_method='layer'`、mmse：前者 sim 段错误、后者超时（>2.5h）不可行。
- **根因定位**（`RESULT-head-hybrid-fp16.md` §5）：box 损伤来自上游（backbone/neck）int8 **激活（a8）量化**累积——box 回归通道是无界 raw 值，对激活噪声敏感；sigmoid 后的 obj/cls 不敏感。只动 head 权重精度无法修复。
- **fp16 参照**：全 fp16 模型忠实（box 0.0100 / 解码 IoU 0.9965，`RESULT-standalone-harness.md`），但**引擎喂不进**：零拷贝路径强制 `input_native.type = RKNN_TENSOR_UINT8`（`hybrid_rga_rknn.cpp:171`），fp16 输入模型与 1 字节/元素的 native 输入不兼容。这也解释了设备 gate 排序里 fp16 反而居中的现象（gate 天花板由测量方法主导，见 §4）。
- **过程中被证伪的假设**（如实记录）：`report-base1-gate2-faithfulness.md` 初版假设「std 255→1 转换错误是根因」；`report-base1-rknn-reconvert.md` 实测重转后 22 帧检出**逐位一致**（引擎对 model_sha256 强校验，`rknn_backend.cpp:254-260`，加载的确实新文件）→ 该假设在设备层面证伪（引擎 native 输入路径下 std 折叠使两种 int8 模型行为等价）。

## 3. B1 recipe（可复用）：int8 输入 + 近全 fp16 激活的混合量化

来源：`report-rknn-b1-input-int8-w8a16.md`。

- **配置**：hybrid quantization；`custom_quantize_layers` 把除 `images` / `images_int8` / 首层 conv 输出 `503` 之外的**全部 202 个可标层设为 float16**（共 205 个量化张量），**保留输入 INT8（zp=-128, scale=1.0）**——即 uint8 画布按 u8↔s8 平移零成本直通，1 字节/元素，引擎零拷贝可喂。
- **实测**（radxa `rknn_dump`，同画布 vs `outA_onnx.npy`）：
  - input attr：`RKNN_TENSOR_INT8, NHWC, zp=-128, scale=1.000000, size=519168` ✓；output FLOAT16；
  - box(0-3) mean err = [0.0167, 0.0154, 0.0130, 0.0121] → **平均 0.0143 ≤ 0.02 ✓**；
  - 解码 dets ref=5 dev=5，**meanIoU = 0.9814（min 0.9745）≥ 0.95 ✓**；
  - 决定性实验中引擎直接喂该模型并输出与 harness 逐位一致的结果（§1）→ **引擎可喂且忠实**。
- **构建要点与坑（照实写）**：
  - 工具链：rknn-toolkit2 **2.3.2**（spark `$WORK_DIR/.venv-rknn`；构建脚本 `/tmp/hqB_build.py`，基于 step1 产物 `/tmp/hqA/{yolox_tiny.model,yolox_tiny.data,cfg_default.bak}`）。
  - **两步构建**：先 step1（w8a8 全量化，calib2 dataset，OL2）拿到默认 quant cfg，再改 `custom_quantize_layers` 后 step2 + export。
  - **concat 张量不能单独标**：直接标 concat 输出会报 `quantize_parameters['823']['scale'] is not allowed to be modified`，必须**连同其 sigmoid 输入一起标**（`RESULT-head-hybrid-fp16.md` §2 A2 的教训，B1 近全 fp16 天然规避）。
  - **模拟器对量化模型推理确定性段错误**（`rknn/api/rknn.py:314`，aarch64 spark，`quant-int8-fix-report.md` blocker A/B）：主线程必崩；缓解 = 在 256MB 栈的 Python 线程里跑完整 build→init_runtime→inference 生命周期（3/3 成功，退出时 teardown 段错误无害）；hybrid step2 路径模型在线程内仍崩且 step2 必抛 `KeyError: '844'`（toolkit 内部 bug），**step2 须主线程跑、推理用线程**——或直接用设备侧 `rknn_dump` harness 验证（推荐，B1 即如此）。
  - toolkit 警告（记录，不影响结果）：input dtype float32→int8、output float32→float16 的提示，以及 `E RKNN: Unkown op target: 0` ×2（step2 仍 ret=0，导出成功）。
  - dataset.txt 必须是 build host 可见的绝对路径（否则报 `The image of /calib/001.jpg is invalid!`）。
- **gate 注意**：B1 离线达标但当时逐流 gate 仍 0.815/0.821 FAIL——**最佳假设（best-supported hypothesis）**是该天花板由两侧解码栈画布差异主导（§4），不是 B1 模型问题。注意该假设尚未被独立证实：决定性实验对比的是 RK 画布 vs ffmpeg 重建（libav 下界估计），**不是 float 消费者真实输入画布**（float 侧此刻无画布 dump 钩子，见 §6-3），且来源报告中源帧（F0/F1）存在歧义；它可确证的是「两侧画布确有像素差」与「同一画布下引擎 ≡ harness」，画布差是否为历史 gate 失败的直接成因仍属推断，待验证。

## 4. gate 2 口径修正：改为「同一画布对比」

- **为什么旧口径不可用**：逐流双消费者 `--faithfulness`（float-on-device vs RK，各自解码 RTSP 流）混入了两侧**解码栈画布差**。决定性实验实测（`report-engine-row0-vs-harness.md` §4）：MPP+RGA 画布 vs avdec/libswscale 画布逐像素差 **MAE 1.33–1.49（/255）、最大差 114、~37% 像素不同、>16 差异占 ~1%，差异集中在检出区**（bbox y[91,324] x[66,363]）。这个量级足以移动弱框/改变检出数，与「内容帧 FAIL、灰帧两侧一致」的 gate 失败模式吻合。其 0.82–0.86 的稳定天花板（fp16 0.8559 / B1 0.8211 / int8 0.7232，`report-rknn-b1-input-int8-w8a16.md` §7 对照表）**与画布差异的存在相容，最佳假设是测量地板而非模型/引擎问题**——但需注意：该实验对比的是 RK 画布 vs **ffmpeg 重建**，不是 float（gst_source）消费者的真实输入画布，且源帧归属（F0/F1）存在歧义，故「画布差异是历史 gate 天花板的成因」目前是**最佳假设，尚非决定性证明**。要证实需要：对比两侧消费者**真实且帧对齐的输入画布**——float 侧（gst_source）此刻**没有画布 dump 钩子**（见 §6-3），要么加钩子（属引擎改动），要么用等价方法构造 float 侧画布并证明其与引擎输入同帧同变换。可确证的部分仅为：上述逐像素差实测存在且集中于检出区；同一画布下引擎行 0 vs RKNN-harness 逐位一致（IoU 1.0000，§1）。
- **新 gate 2 步骤模板**（已在决定性实验中完整跑通）：
  1. **引擎 dump 画布**：radxa 启动引擎 `VB_RK_DUMP_CANVAS=/out/…/canvas.raw`（dump 在第一次 `rknn_run` 后、`rknn_outputs_get` 前写画布 ⇒ 与 parity 行 0 同帧，`hybrid_rga_rknn.cpp:335-380`），`--frames 1` 得 1 行 parity.jsonl；
  2. **spark onnxruntime**：同一 canvas.raw（416×416×3，BGR 0-255）喂 `yolox_tiny.onnx`，得 float 参考 raw；
  3. **radxa `rknn_dump`**：同一 canvas.raw int8 native 喂 B1（或待测）.rknn，得设备 raw；
  4. **同一套引擎语义解码**：strides 8/16/32、row-major grid decode、obj×cls、thr 0.3、贪心 class-aware NMS 0.45（对齐 `core-cpp/vb/src/post/{decoder,yolox_decode,nms}.cpp`）；参考实现 `/tmp/row0_compare.py`（spark）。
- **判据**（沿用离线口径并补齐数量/空结果项，与 `tools/vb_parity_compare.py` 实际检查项一致）——**全部满足**才算通过：
  1. box(0-3) mean err **≤ 0.02**；
  2. 解码 meanIoU **≥ 0.95**；
  3. **未匹配的参考检出数 = 0**（即每帧每个参考检出都必须配对，`vb_parity_compare.py`：`unmatched > 0` 即 FAIL；如需放宽须给出明确容忍上限，不得默认）——这条防止靠丢检出让剩余配对满足 1/2 的假通过；
  4. **每帧数量差 ≤ 1**（沿用 `vb_parity_compare.py` 的 `--count-diff` 语义，默认 1）；
  5. **空结果处理**（显式规则，不得因「没有可配对的框」静默通过）：参考 0 检出而设备 >0 检出 → **FAIL**（数量差超限）；两侧都 0 → 记为该帧通过但**计入覆盖统计**（报告零检出帧占比，避免整段空流以 0/0 meanIoU 静默达标）。注意：`vb_parity_compare.py` 当前对「两侧都 0」的帧仅自然通过、未单列覆盖统计，新口径要求统计侧补充这一项（工具改动另行立项，文档先立口径）。
  - B1 已达标（box 0.0143 / meanIoU 0.9814），引擎行 0 同口径复现 0.9814。
- 逐流双消费者 `--faithfulness` 降级为**端到端诊断指标**，不得作为转换保真度判定。

## 5. parity 夹具的使用与边界

提交链：`8d08bf4`（对齐夹具）→ `7a12eb6`（内容相位校验）→ `ed40553`（修 tpad/时间戳回归）。协议见 `tests/fixtures/parity/README.md` 与 `report-parity-fixture-phase-check.md`。

- **使用**：`tools/parity_feed.sh` 单次发布 + join 窗口（定版参数 `--fps 5 --window 20`；默认 8s 会因消费者 open 重试超时 exit 4）；`--pattern odd-empty` 渲染时交错灰帧（0x727272）使检出数序列成为已知模式；`tools/parity_fixture_check.py --expect-pattern odd-empty` 做内容相位校验。
- **职责边界**（README 原文语义）：**check `exit 3` = 夹具未对齐**（行数≠N 或相位错位）——此时 compare 数字**作废**，必须重跑实验，不得调阈值；**compare `exit 1` = 模型分歧**——仅在 check `exit 0` 前提下才算数。一句话：check 管「两侧看到的是不是同一帧序列」，compare 管「同一帧上两个模型的行为差多少」。
- **已验证的能力**：真机抓到 gst_source 丢首帧 bug（float 侧 21/21 行整但整体左移 1，旧行数校验放行、新校验 exit 3，`report-parity-fixture-phase-check.md` §2）；单测 22 passed。
- **已知局限**（如实声明）：纯交替模式只能抓**奇数**总偏移（含 +1），偶数偏移 ≥2 在空/非空意义上不可区分；更强指纹需引擎输出帧号（改引擎，未做）。
- 注：夹具使用过程曾受消费者侧问题干扰——float（gst_source）消费者**确定性丢 payload 帧 0**、radxa RK 经 VPN 拉流时曾系统性丢灰帧（见该报告 §3b）。

## 6. 待立项的引擎问题（逐条，含证据）

1. **`gst_source` 确定性丢 payload 首帧、无发布者时仍产出垃圾帧**
   - 丢首帧：g9/g10 float 序列 = 正确序列左移 1（`report-base1-rknn-reconvert.md` 发现 1）；相位校验真机复现 3/3 翻转（`report-parity-fixture-phase-check.md` §2/§3b）。
   - 垃圾帧：无任何发布者时 float（gst_source）消费者仍产出 22 帧「垃圾帧」（0 检出、~9fps，RTSP 404 重试循环中产生）；RK（mpp_source）同条件正确报 0 帧。怀疑 rtspsrc 反复 open 失败时 appsink 发出未初始化/空缓冲（`report-base1-gate2-faithfulness.md` 排障记录 2）。
2. **零拷贝输入固定 UINT8 → fp16 输入模型不可用（限制记录）**：`hybrid_rga_rknn.cpp:171` 强制 `input_native.type = RKNN_TENSOR_UINT8`；全 fp16 模型忠实（0.0100/0.9965）但引擎无法以零拷贝喂入。B1 recipe 即绕开此限制的方案；若未来要支持 fp16 输入模型需改此处。另相关：设备与模拟器在 w8a8 + zero-copy uint8 native 输入路径上行为不一致（两个 std 不同的 int8 模型模拟器输出不同、设备逐位相同，`report-base1-rknn-reconvert.md` 发现 2）。
3. **float 侧无画布/输出 dump 钩子**：`VB_RK_DUMP_CANVAS` 只覆盖 RK 引擎路径；决定性实验中 float 对照画布只能用 ffmpeg CLI 重做（libav 下界估计，RGA NV12 色度量化与 bilinear 细节未逐项复刻，`report-engine-row0-vs-harness.md` §4 局限）。给 float 路径加同款 dump 钩子可消除这一估计误差。

## 7. 复现清单（最少命令；路径均为实际存在的工具/产物）

环境：spark（构建/ONNX 参考）、radxa rock-5t（RK3588 设备）、fleet 镜像 `vb-base-rk:m21-head7740d03`；toolkit 2.3.2 venv `$WORK_DIR/.venv-rknn`（spark）。

```bash
# 1. 离线参考（spark）：ONNX float 参考 raw
#    画布 canvas416.raw (sha256 4a2e80fd…) + outA_onnx.npy [3549,85]
#    产出：$WORK_DIR/offline-ab/outA_onnx.npy

# 2. B1 模型构建（spark，两步 hybrid；脚本 /tmp/hqB_build.py，基于 /tmp/hqA step1 产物）
#    custom_quantize_layers: 保留 {'503','images','images_int8'} int8，其余 202 层 float16
#    产出：$WORK_DIR/offline-ab/yolox_tiny_B1_firstconv_int8.rk3588.rknn
#          sha256 6c01d19eecdb3e4349e173ada2499e9010858c197f54957084e476cbab53d447

# 3. 设备侧独立 harness（radxa）
gcc -O2 -o rknn_dump rknn_dump.c -lrknnrt   # offline-ab/rknn_dump.c
./rknn_dump $WORK_DIR/models/yolox_tiny_B1_firstconv_int8.rk3588.rknn \
            $WORK_DIR/out/canvas416.raw out_b1
#    预期 input: INT8 NHWC zp=-128 scale=1.0 size=519168（引擎零拷贝可喂）
#    对比 outA_onnx.npy：box(0-3) mean 0.0143 / 解码 meanIoU 0.9814

# 4. 决定性实验：引擎行 0 vs 同画布两路（gate 2 新口径模板）
radxa$ cd $WORK_DIR && VB_M21_IMAGE=vb-base-rk:m21-head7740d03 \
  VB_RK_DUMP_CANVAS=/out/row0-check/canvas.raw \
  nohup ./dump-run.sh --parity /out/row0-check --frames 1 --config /config/parity-fixture-rk.json &
spark$ tools/parity_feed.sh $WORK_DIR/media/vb-720p15-h264.mp4 --fps 5 --readers 1 --window 20
radxa$ cd offline-ab && ./rknn_dump <B1.rknn> $WORK_DIR/out/row0-check/canvas.raw out_row0check
#    spark onnxruntime 同 canvas → /tmp/row0_compare.py 解码（strides 8/16/32, obj×cls,
#    thr 0.3, class-aware NMS 0.45）
#    预期：引擎行0 vs RKNN-harness 6/6 框 IoU 1.0000；vs ONNX meanIoU 0.9814

# 5. 逐流 gate（现降级为诊断）：夹具 + 相位校验 + compare
spark$ tools/parity_feed.sh $WORK_DIR/media/vb-720p15-h264.mp4 --fps 5 --readers 2 --window 20 --pattern odd-empty
spark$ uv run python tools/parity_fixture_check.py --ref <rk>/parity.jsonl --got <float>/parity.jsonl \
         --frames <N> --expect-pattern odd-empty     # exit 0 才继续
spark$ uv run python tools/vb_parity_compare.py --ref <float>/parity.jsonl --got <rk>/parity.jsonl \
         --faithfulness --threshold 0.95              # 诊断口径，预期卡 0.82–0.86（测量地板）
```

## 8. 报告间矛盾说明

- `report-base1-gate2-faithfulness.md` 的「std 255→1 是根因」假设与 `report-base1-rknn-reconvert.md` 的证伪结果矛盾 → **以后者为准**（重转后设备检出逐位一致，引擎 sha256 强校验排除加载错文件）。
- `report-base1-rknn-reconvert.md` 的「fp16 输入位型错乱导致 gate 偏低」单因解释与 `report-rknn-b1-input-int8-w8a16.md` 的 gate 排序矛盾 → 引擎路径逐位可信、B1 模型忠实可由 `report-engine-row0-vs-harness.md` 直接确证；gate 天花板的成因以该报告的**最佳假设**为准（两侧画布差异主导），但如 §4 所述，画布差 ↔ 历史 gate 失败的因果链还需 float 侧真实画布对照才能坐实，标注为 best-supported hypothesis、待验证。
- `report-parity-feed-regression-fix.md` 文件缺失，其对应改动以提交 `ed40553`（修 parity_feed 回归——恢复 tpad、pattern 改按帧序号定时）为准。
