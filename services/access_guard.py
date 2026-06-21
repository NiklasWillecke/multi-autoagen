import os
import secrets
from typing import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "").strip()
COOKIE_NAME = "access_token"
UNAUTHORIZED_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>Zugang geschützt</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #111; color: #eee;
           display: flex; align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; }
    main { max-width: 28rem; padding: 2rem; text-align: center; }
    h1 { font-size: 1.25rem; margin-bottom: 0.75rem; }
    p { color: #aaa; line-height: 1.5; }
    code { background: #222; padding: 0.15rem 0.4rem; border-radius: 4px; }
  </style>
</head>
<body>
  <main>
    <h1>Zugang geschützt</h1>
    <p>Diese Instanz ist privat. Bitte den Link mit
    <code>?token=...</code> verwenden, den du erhalten hast.</p>
  </main>
</body>
</html>"""


def access_protection_enabled() -> bool:
    return bool(ACCESS_TOKEN)


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() or None
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return cookie
    query = request.query_params.get("token")
    if query:
        return query
    return None


def _token_valid(token: str | None) -> bool:
    if not token or not ACCESS_TOKEN:
        return False
    return secrets.compare_digest(token, ACCESS_TOKEN)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept and request.url.path in {"/", "/index.html"}


def _unauthorized_response(request: Request) -> Response:
    if _wants_html(request):
        return HTMLResponse(UNAUTHORIZED_HTML, status_code=401)
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


def _attach_access_cookie(response: Response, token: str) -> None:
    secure = os.getenv("ACCESS_TOKEN_SECURE_COOKIE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=secure,
        max_age=60 * 60 * 24 * 30,
    )


class AccessGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not access_protection_enabled():
            return await call_next(request)

        token = _extract_token(request)
        if not _token_valid(token):
            return _unauthorized_response(request)

        response = await call_next(request)
        if request.query_params.get("token"):
            _attach_access_cookie(response, token)
        return response
