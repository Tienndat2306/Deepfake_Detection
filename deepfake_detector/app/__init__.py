"""Flask app factory for the deepfake detector web UI."""

from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,
        UPLOAD_FOLDER=str(app.root_path + "/static/uploads"),
    )

    from .routes import bp

    app.register_blueprint(bp)
    return app
