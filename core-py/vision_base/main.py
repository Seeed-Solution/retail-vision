"""Entry point: ``python -m vision_base.main --config <vb.config/1> [--validate]``.

Loads the config, optionally validates only, or runs the Supervisor until
SIGINT/SIGTERM. App modules are resolved from the config file's directory
(§5.5.2 instantiation rule). Standard library only.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision_base.main")
    parser.add_argument("--config", required=True,
                        help="path to a vb.config/1 JSON file")
    parser.add_argument("--validate", action="store_true",
                        help="validate the configuration and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from .config import load
    try:
        cfg = load(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.validate:
        print(f"ok: {cfg.device_id} ({len(cfg.streams)} streams)")
        return 0

    from .supervisor import Supervisor
    sup = Supervisor(cfg)

    def _shutdown(signum, frame):
        sup.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    sup.start()
    sup.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
