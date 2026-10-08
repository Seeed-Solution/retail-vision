"""Real native CPU/JPEG protocol checks; run with VB_STAGE2_RUNTIME_BIN."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from vision_base.runtime_client import RuntimeClient, RuntimeError_, RuntimeGone


@pytest.fixture
def stage2_setup(tmp_path):
    binary = os.environ.get("VB_STAGE2_RUNTIME_BIN")
    if not binary:
        pytest.skip("VB_STAGE2_RUNTIME_BIN must select a CPU+JPEG native build")
    import onnx
    from onnx import TensorProto, helper
    # Seven timesteps decode independently to ABBC: A,A,blank,B,blank,B,C.
    symbols = [1, 1, 0, 2, 0, 2, 3]
    logits = [8.0 if symbol == c else -8.0 for c in range(4) for symbol in symbols]
    tensor = helper.make_tensor("logits", TensorProto.FLOAT, [1, 4, 7], logits)
    graph = helper.make_graph([helper.make_node("Constant", [], ["out"], value=tensor)],
                              "stage2-protocol", [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 3, 2, 2])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 7])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "stage2.onnx"
    onnx.save(model, path)
    config = {"backend": {"name": "synthetic"}, "stage2": {
        "backend": "cpu", "model_path": str(path), "input_hw": [2, 2],
        "charset": ["", "A", "B", "C"], "roi": None}}
    def client(config=config):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        return RuntimeClient([binary, "--ipc-fd", "{fd}"], str(config_path),
                             on_frame=lambda *a: None, on_event=lambda *a: None,
                             on_stats=lambda *a: None, on_state=lambda *a: None, on_exit=lambda *a: None)
    def jpeg(w, h):
        return subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                               f"color=red:s={w}x{h}", "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                              check=True, stdout=subprocess.PIPE).stdout
    return client, config, jpeg


def test_native_infer_image_jpeg_and_parallel_requests(stage2_setup):
    factory, _, jpeg = stage2_setup
    c = factory()
    try:
        assert c.start().stage2_ready
        for w, h in [(2, 2), (8, 4), (4, 8)]:
            result = c.infer_image(jpeg(w, h), timeout_s=5)
            assert (result["w"], result["h"], result["text"]) == (w, h, "ABBC")
            assert .999 < result["min_char_conf"] <= result["mean_conf"] <= 1
            assert result["infer_ms"] >= 0
        image = jpeg(4, 4)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: c.infer_image(image, timeout_s=5), range(8)))
        assert [r["text"] for r in results] == ["ABBC"] * 8
        for image in [b"not-jpeg", b"\xff\xd8\xffbroken"]:
            with pytest.raises(RuntimeError_, match="bad jpeg"):
                c.infer_image(image)
        with pytest.raises(RuntimeError_, match="base64"):
            c.request("infer_image", 2, jpeg_b64="!!!!")
        with pytest.raises(RuntimeError_, match="image too large"):
            c.infer_image(b"x" * (1024 * 1024 + 1))
    finally:
        started = time.monotonic()
        c.stop()
        assert time.monotonic() - started < 1
        assert c.proc.returncode == 0


def test_native_stage2_startup_rejects_charset_mismatch(stage2_setup):
    factory, config, _ = stage2_setup
    config["stage2"]["charset"].pop()
    c = factory(config)
    with pytest.raises(RuntimeGone):
        c.start()
    assert c.proc.poll() is not None


def test_native_cpu_backend_registration_is_linked(stage2_setup):
    factory, config, _ = stage2_setup
    config["backend"] = {"name": "cpu", "model_path": config["stage2"]["model_path"]}
    c = factory(config)
    try:
        hello = c.start()
        assert hello.backend == "cpu"
        assert hello.model_hw == (2, 2)
        assert hello.stage2_ready
    finally:
        c.stop()
        assert c.proc.returncode == 0


def test_native_without_top_stage2(stage2_setup):
    factory, _, jpeg = stage2_setup
    c = factory({"backend": {"name": "synthetic"}})
    try:
        assert not c.start().stage2_ready
        with pytest.raises(RuntimeError_, match="stage2 not configured"):
            c.infer_image(jpeg(2, 2))
    finally:
        c.stop()
