"""Run one OSGym risk Flask app on a supervisor-selected port."""

import argparse
import importlib
import logging
import os
import uuid

from flask import jsonify


HEALTH_PATH = "/__osgym_health__"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--port", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    module = importlib.import_module(args.module)
    app = getattr(module, "app")
    instance_id = uuid.uuid4().hex

    def health():
        return jsonify(
            {
                "service": args.service,
                "module": args.module,
                "port": args.port,
                "pid": os.getpid(),
                "instance_id": instance_id,
            }
        )

    app.add_url_rule(
        HEALTH_PATH,
        endpoint="osgym_risk_service_health",
        view_func=health,
        methods=["GET"],
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
