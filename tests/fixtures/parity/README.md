# parity 帧源对齐夹具（BASE-1）

目的：让 `vb-runtime --parity` 的两端（CPU 参考 / 平台后端）看到**同一帧序列**，使
`vb_parity_compare.py` 的逐行配对（line i ↔ line i）判定有效。当前 `parity.jsonl`
无帧号、无内容指纹，纯按行配对；只要帧源不对齐（例如旧的 `-stream_loop -1`
循环源，两端从不同相位加入），失败就无法与模型分歧区分。

组成：

| 文件 | 作用 |
| --- | --- |
| `tools/parity_feed.sh` | 预渲染限速全 I 帧片段（doorbell + join 窗口 + warm-up gap，见下），起独立 mediamtx，**单次**发布，打印 `PARITY_FRAMES=<N>` |
| `tools/parity_fixture_check.py` | 校验两侧 `parity.jsonl` 各恰好 N 行；不等 → exit 3（夹具失败，不是模型分歧） |
| `tools/fixtures/parity/parity-fixture-*.json` | 消费者示例 config（fixture 端口 18664、`open_timeout_s: 120`、`reconnect_delay_s: 0.2`） |
| `tests/test_parity_fixture_check.py` | check 工具单元测试 |

## 时间线（发布端编排）

发布的片段（全 I 帧 `keyint=1`，`fps=--fps`）时间戳被改写为：

```
F0  @ t=0        doorbell：第一个 RTP 包，让 -re 按增量把 join 窗口走完。
                 不计入 payload。
F1  @ t=WINDOW   第一帧 payload（WINDOW 默认 8s）。
F2  @ t=WINDOW+WARMUP   （WARMUP 默认 5s）：消费者对 F1 做一次性 warm-up
                 （解码器/后端首帧初始化，~0.5-2s）期间没有后续帧到达，
                 drop=true 的 appsink 不会丢掉计数帧。
F3..             之后连续按 --fps 播完，只播一遍。
结尾 6 帧克隆    单遍流收尾的 RTP 帧在 publisher teardown 时经常丢失；
                 消费者collect到 --frames N 就停，克隆帧不会进入对齐的
                 parity.jsonl。
```

消费者 attach 时序：mediamtx（`source: publisher`）在 path 未 ready 时对读者
回 404，读者无法"等"发布者；但 `vb-runtime` 会在 add deadline
（`runtime.open_timeout_s`，需 ≥120）内持续重试 open。path 一旦 ready，重试
循环 ~0.2s 内即可 attach。引擎的 open() 在**收到第一个 buffer** 时才完成，
单次尝试有 ~5s 的状态超时，超时后重建连接（周期 ~5.4s）。因此 W=8 保证：
消费者 attach 后 ≤5.4s 内必然有一次尝试落在 (W-5, W) 区间、在 F1 到达时完成
open，从而从 F1 开始完整采集。

## 运行协议（顺序很重要）

1. **先启动消费者**（它们会持续重试 open）：
   - <dev-host>：`vb-runtime --config tools/fixtures/parity/parity-fixture-cpu-ref.json --parity <cpu_dir> --frames <N>`
   - <rk3588-board>：`vb-runtime --config tools/fixtures/parity/parity-fixture-rk-rad.json --parity <rk_dir> --frames <N>`
2. 运行 `tools/parity_feed.sh <clip> --fps 5 --readers 2`。
   脚本打印 `CONSUMERS_NOW` 表示 path 已 ready（若消费者尚未启动，此时启动
   也来得及，但预启动更稳）。
3. feed 退出 0 后，等待 ≥30s（引擎的 parity 收集有 30s 上限周期），
   两侧各得到 `<dir>/parity.jsonl`。
4. `uv run python tools/parity_fixture_check.py --ref <cpu_dir>/parity.jsonl --got <rk_dir>/parity.jsonl --frames <N>`
   必须退出 0（N 用 feed 打印的 `PARITY_FRAMES`；消费者 `--frames` 也用这个 N）。
   若 feed 用了 `--pattern odd-empty`，把打印的 `PARITY_PATTERN` 传给 check：
   `... --expect-pattern odd-empty`（见下节「内容相位校验」）。旧行为（只查行数）
   保留为 `--frames-only`。
5. check 通过后 `vb_parity_compare.py` 的结论才算数：
   - `exit 0` = 通过；`exit 1` = 模型分歧（真实信号）；
   - check `exit 3` = 夹具未对齐（丢帧/晚 attach/**内容相位错位**）→ **重跑**，不要调阈值。

## 职责边界：parity_fixture_check.py vs vb_parity_compare.py

- **check 的 `exit 3` = 夹具未对齐**：两侧 parity.jsonl 的行配对无意义——行数不等于
  N（丢帧/晚 attach），或内容相位错位（丢首帧等，见下节）。此时 compare 的任何
  数字都作废，必须重跑实验，**不得**调 compare 阈值。
- **compare 的 `exit 1` = 模型分歧**：仅在 check `exit 0` 之后，逐行配对才有意义，
  此时 compare 失败是真实的模型行为差异信号。
- 一句话：check 管的是「两侧看到的是不是同一帧序列」，compare 管的是「同一帧上
  两个模型的行为差多少」。check 不做模型判定，compare 不做夹具判定，退出码语义
  互不重叠。

## 内容相位校验（--expect-pattern）

**为什么行数不够**（实测，2026-10 <rk3588-board>）：float（gst_source）消费者稳定丢失
payload 第 0 帧，其 22 行序列 = 正确序列左移 1 帧——行数仍是 N，旧行数校验依然报
ALIGNED，导致下游逐行配对全部带 +1 相位污染。parity.jsonl 每行只有 detections，
无帧指纹，所以相位证据由夹具发布端制造：

- `parity_feed.sh --pattern odd-empty` 在 payload 中交错插入纯灰帧（0x727272，
  与画布 letterbox 填充同值，无检出），使两侧检出数序列成为已知模式：
  **偶数行（line 0,2,4,…）有检出，奇数行为 0**（payload 第 0 帧是内容帧）。
  feed 打印 `PARITY_PATTERN=odd-empty`。
- check 传 `--expect-pattern odd-empty` 后按行校验该模式。**奇数总偏移
  （含丢首帧的 +1）必然使相位翻转 → exit 3**，并输出偏移结论
  （`shifted by +1 (first payload frame(s) dropped ...)`）。
- 相位校验按侧独立进行，**不要求两个模型的检出数一致**——模式只区分
  「空 / 非空」，模型分歧不影响判定。也接受逗号分隔的精确检出数列表。
- 局限（如实声明）：纯交替模式只能抓**奇数**偏移；偶数偏移（≥2）与正确相位在
  空/非空意义上不可区分。实测的 bug 是 +1，可抓。更强的指纹需要引擎输出帧号
  （改引擎，超出本夹具范围）。
- 要求源片段的每个内容帧在该模型 score 阈值下都 ≥1 检出（BASE-1 的
  vb-720p15-h264.mp4 实测 float/RK 每帧 1–6 检出，满足）。

## fps / window / warmup 选择规则

- `FPS ≤ 1 / (最慢消费者单帧 p95 耗时 × 2)`，先 5 fps 起步。全 I 帧保证每帧
  独立解码、耗时均匀。
- `WINDOW`（默认 8）需 > 引擎单次 open 状态超时（5s）且 ≥ 消费者 attach 延迟
  + 余量；`WARMUP`（默认 5）需 > 最慢消费者首帧 warm-up 时间。
- 参数变化会改变渲染缓存的 key（`/tmp/parity-feed-cache/`），首次运行会重新
  渲染（~5s），之后的重跑 publisher 启动零抖动。

## 退出码语义

| 退出码 | 来源 | 含义 |
| --- | --- | --- |
| 0 | feed / check | 夹具 OK（发布完成 / 对齐校验通过） |
| 1 | feed | 用法错误；`vb_parity_compare.py` 的 1 = 模型分歧 |
| 2 | feed / check | ffmpeg/ffprobe 失败（feed）；用法或 IO 错误（check） |
| 3 | feed / check | 环境失败：mediamtx 没起来（feed）；**夹具未对齐/丢帧**（check） |
| 4 | feed | 读者在 join 窗口内未到齐（重跑；持续失败按上面规则调 WINDOW/WARMUP/FPS） |

`vb_parity_compare.py` 的判定语义与退出码不在本夹具职责内，一律不改。

## 失败要重跑，而不是调阈值

行数不齐（check exit 3）说明有消费者晚 attach 或丢帧——夹具/环境问题。
正确反应：原样重跑一次；连续两次失败再调 `--window`/`--warmup`/`--fps`。
**禁止**为了让 gate 通过而放宽 `--iou` / `--count-diff`。

## 禁止事项

- 不要再用 `-stream_loop -1` 的连续循环源做量化 parity 判定（对照实验实测：
  两个同机消费者起始帧错 7s 时 `vb_parity_compare.py` 18/22 帧 FAIL、exit 1）。
- 不要动 <dev-host> 上既有的 `:8664` mediamtx 实例与 `$WORK_DIR/config/*` 既有文件；
  夹具用自己的端口（默认 RTSP 18664 / API 19970，TCP only）和
  `tools/fixtures/parity/` 下的 config。

## mediamtx 版本兼容性（重要）

- **必须用 mediamtx v1.12.3**（`$WORK_DIR/bin/mediamtx-1.12.3`，脚本默认）。
  v1.21.1 对 vb 基础镜像里的 GStreamer 1.22 rtspsrc 会回 **400 Bad Request**
  （`can't get sdp`，消费者永远打不开流）；v1.12.3 实测正常。
- API 路径 `GET /v3/paths/list` 的 `items[].readers`（数组）已按 v1.12.3 核对。
- 脚本已把 `readTimeout`/`writeTimeout` 提到 150s（默认 10s 会在静默期掐断
  publisher），并只开 TCP（既有 :8664 实例占着 UDP 8000/8001 与 RTMP 1935）。

## 已知残余问题 / 后续

- **<rk3588-board> 侧本轮未完成端到端验收**：<rk3588-board> 上的 `vb-base-rk:m21` 容器消费者
  （MPP 源）对任何 RTSP 源（包括旧 loop 源与夹具源）都采集 0 帧且不打印任何
  open 错误——与夹具无关，属设备侧环境回退（上一轮 M2.1 的镜像/环境曾被验收
  过）。宿主 gst-launch 直连夹具 path 是正常的（能 attach），说明网络与
  mediamtx 均无问题。下一步：核对镜像内 vb-runtime 与 mpp 源的版本/日志
  （`GST_DEBUG=3`）、`/dev/dri`、librga 版本后再跑 gate 1/gate 2。
