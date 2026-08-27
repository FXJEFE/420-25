# -*- coding: utf-8 -*-
"""
Pipeline-safe waitress entry.
Default: one-shot health check (does not hang pipeline).
Use --serve to actually start waitress on api_port.
"""
from fxjefe_paths import load_config, setup_logging
import logging
import sys

config = load_config()
setup_logging(config, "waitress_server")


def main() -> None:
    if "--serve" not in sys.argv:
        # pipeline mode: verify ai_server import + optional health
        try:
            import ai_server  # noqa: F401
            logging.info("ai_server module import OK")
        except Exception as e:
            logging.warning("ai_server import: %s", e)
        try:
            import requests
            url = (config.get("ai_server_url") or "http://127.0.0.1:8080").rstrip("/") + "/health"
            r = requests.get(url, timeout=3)
            logging.info("health %s %s", r.status_code, r.text[:300])
        except Exception as e:
            logging.info("health check skipped/failed (start ai_server.py separately): %s", e)
        logging.info("waitress server.py pipeline check complete (use --serve to run server)")
        return

    # real serve mode
    from waitress import serve
    from ai_server import app

    host = "0.0.0.0"
    port = int(config.get("api_port") or 8080)
    logging.info("Starting waitress on %s:%s", host, port)
    serve(app, host=host, port=port)


if __name__ == "__main__":
    main()
