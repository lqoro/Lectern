"""Entry point PyInstaller freezes into Lectern.app's bundled server.

Same as `python -m lectern`, which a frozen executable cannot be asked for.
"""
from lectern import config
from lectern.__main__ import main

if __name__ == "__main__":
    config.ensure_dirs()
    raise SystemExit(main())
