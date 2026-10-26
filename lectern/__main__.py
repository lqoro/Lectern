"""Entry point: `python -m lectern`."""
from __future__ import annotations

import logging
import sys

from waitress import serve

from . import config
from .server import create_app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(config.LOG_DIR / "lectern.log")]
        if config.LOG_DIR.exists() else None,
    )
    app = create_app()
    print(f"\n  {config.APP_NAME} {config.VERSION}", file=sys.stderr)
    print(f"  Lectures : {config.LECTURES_ROOT}", file=sys.stderr)
    print(f"  Open     : http://127.0.0.1:{config.PORT}/", file=sys.stderr)
    if not config.is_loopback_only():
        ip = config.lan_ip() or config.HOST
        print(f"  On phone : http://{ip}:{config.PORT}/?k={config.access_token()}",
              file=sys.stderr)
        print("             (network mode: this link is the key -- treat it "
              "like a password)", file=sys.stderr)
    print("", file=sys.stderr)
    # threads>1 matters: audio streaming with Range must not block the API
    # that the recording indicator polls.
    # max_request_body_size: waitress refuses bodies over 1 GB by default,
    # which a long lecture video easily exceeds (see /api/import).
    serve(app, host=config.HOST, port=config.PORT, threads=8,
          ident="Lectern", clear_untrusted_proxy_headers=True,
          max_request_body_size=16 * 1024**3)
    return 0


if __name__ == "__main__":
    config.ensure_dirs()
    raise SystemExit(main())
