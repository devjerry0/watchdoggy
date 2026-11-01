from doggy.web.routers import wifi

IW_LINK = """Connected to 04:42:1a:d9:87:d4 (on wlan0)
\tSSID: potatochip
\tfreq: 5500.0
\tRX: 463875194 bytes (1993966 packets)
\tsignal: -57 dBm
\trx bitrate: 234.0 MBit/s
"""


class _Done:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_link_parses_iw_output(monkeypatch):
    monkeypatch.setattr(wifi.subprocess, "run", lambda *a, **k: _Done(0, IW_LINK))
    assert wifi._link() == {"node": "04:42:1a:d9:87:d4", "ssid": "potatochip",
                            "freq_mhz": 5500.0, "signal_dbm": -57}


def test_link_is_empty_when_disconnected(monkeypatch):
    monkeypatch.setattr(wifi.subprocess, "run", lambda *a, **k: _Done(0, "Not connected.\n"))
    assert wifi._link() == {}


def test_link_is_none_without_iw(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("iw")
    monkeypatch.setattr(wifi.subprocess, "run", boom)
    assert wifi._link() is None


def test_repick_drops_the_request_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    router = wifi.build_router()
    repick = next(r.endpoint for r in router.routes if r.path == "/api/wifi/repick")
    status = next(r.endpoint for r in router.routes if r.path == "/api/wifi")
    monkeypatch.setattr(wifi, "_link", lambda: {})
    assert status()["repick_pending"] is False
    assert repick() == {"ok": True}
    assert (tmp_path / "wifi-repick.request").exists()
    assert status()["repick_pending"] is True
