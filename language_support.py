
from flask import (
    request, session, redirect, url_for,
    make_response
)
from urllib.parse import urlparse

from translations import LANGUAGES, translate


def init_language(app):

    @app.before_request
    def load_language():
        saved_language = request.cookies.get("retina_language", "en")

        if saved_language not in LANGUAGES:
            saved_language = "en"

        if "language" not in session:
            session["language"] = saved_language

        if session.get("language") not in LANGUAGES:
            session["language"] = "en"

    @app.context_processor
    def inject_language():
        language = session.get("language", "en")

        return {
            "current_language": language,
            "languages": LANGUAGES,
            "t": lambda key: translate(language, key)
        }

    @app.route("/set-language", methods=["POST"])
    def set_language():
        selected = request.form.get("language", "en")

        if selected not in LANGUAGES:
            selected = "en"

        session["language"] = selected

        # Return to the same page, but avoid POST-only endpoints.
        next_url = request.form.get("next", "/")

        parsed = urlparse(next_url)

        if (
            parsed.scheme
            or parsed.netloc
            or not next_url.startswith("/")
            or next_url.startswith("//")
        ):
            next_url = url_for("dashboard") if session.get("user_id") else url_for("account")

        response = make_response(redirect(next_url))

        response.set_cookie(
            "retina_language",
            selected,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            secure=bool(app.config.get("SESSION_COOKIE_SECURE")),
            samesite="Lax"
        )

        return response
