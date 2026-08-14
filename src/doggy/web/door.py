"""The onboarding door: a tiny plain-HTTP app on the web port.

A web page cannot install a certificate (the OS security model forbids it),
but a plain-HTTP page *can* probe whether this device already trusts the home
CA -- an http page may fetch an https URL on the same host, and that fetch
fails with a TLS error exactly when the CA is untrusted. So port 8000 stays
plain HTTP forever as a "door": old bookmarks keep working, and the page
either redirects to the real https dashboard (trust already installed) or
walks the visitor through the one-time CA install.

The door is intentionally unauthenticated and serves only public material
(the home CA certificate, which every device is meant to trust). The page
and profile payloads themselves live in `door_content`.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from doggy.web.door_content import door_page, mobileconfig


def create_door_app(settings) -> FastAPI:
    app = FastAPI(title="doggy-door")

    def _ca_path() -> Path:
        if settings.ca_cert and Path(settings.ca_cert).is_file():
            return Path(settings.ca_cert)
        raise HTTPException(status_code=404, detail="not set up")

    @app.get("/", response_class=HTMLResponse)
    def door() -> HTMLResponse:
        _ca_path()  # only meaningful once TLS + CA are set up
        return HTMLResponse(door_page(settings.ssl_port))

    @app.get("/ping")
    def ping() -> Response:
        # Cross-origin trust probe target; also answered on the https dashboard.
        return Response(status_code=204, headers={"Access-Control-Allow-Origin": "*"})

    @app.get("/ca.pem")
    def ca_pem() -> Response:
        return Response(_ca_path().read_bytes(), media_type="application/x-pem-file",
                        headers={"Content-Disposition": 'attachment; filename="watchdoggy-ca.pem"'})

    @app.get("/ca.mobileconfig")
    def ca_mobileconfig() -> Response:
        return Response(mobileconfig(_ca_path()),
                        media_type="application/x-apple-aspen-config",
                        headers={"Content-Disposition":
                                 'attachment; filename="watchdoggy-ca.mobileconfig"'})

    return app
