"""Entry point: `python -m lectern`."""
from __future__ import annotations

import logging
import socket
import sys

from waitress import serve

from . import config
from . import platform_compat as pc
from .server import create_app


def _claim_port() -> socket.socket | None:
    """Bind the port BEFORE the app starts: startup recovery must never run in
    a second copy (a second Lectern.bat double-click, a second app launch),
    where it would "recover" lectures the first copy is still processing."""
    family = socket.AF_INET6 if ":" in config.HOST else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if pc.IS_WIN:
        # Without this, Windows lets a second socket bind a port in use.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((config.HOST, config.PORT))
    except OSError:
        sock.close()
        return None
    return sock


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(config.LOG_DIR / "lectern.log")]
        if config.LOG_DIR.exists() else None,
    )
    sock = _claim_port()
    if sock is None:
        print(f"{config.APP_NAME} is already running at "
              f"http://127.0.0.1:{config.PORT}/", file=sys.stderr)
        return 0
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
    serve(app, sockets=[sock], threads=8,
          ident="Lectern", clear_untrusted_proxy_headers=True,
          max_request_body_size=16 * 1024**3)
    return 0


if __name__ == "__main__":
    config.ensure_dirs()
    raise SystemExit(main())
