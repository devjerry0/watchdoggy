"""scripts/netwatch.sh: escalation ladder, node lock, and re-pick, run against
stub system commands.

Each scenario feeds a sequence of ping results; the ping stub ends the run
(kills netwatch) when the sequence is used up. The nmcli stub keeps the WiFi
profile's BSSID lock in $STUB/lock.
"""
import os
import signal
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "netwatch.sh"
NODE = "aa:bb:cc:dd:ee:ff"     # the node the Pi is on
STRONG = "11:22:33:44:55:77"   # the strongest node broadcasting the profile's SSID
SCAN = "\\n".join([
    f"BSS {NODE}(on wlan0) -- associated", "\\tfreq: 5500.0", "\\tsignal: -60.00 dBm", "\\tSSID: home-net",
    f"BSS {STRONG}(on wlan0)", "\\tfreq: 5500.0", "\\tsignal: -48.00 dBm", "\\tSSID: home-net",
    "BSS 11:22:33:44:55:70(on wlan0)", "\\tfreq: 2412.0", "\\tsignal: -33.00 dBm", "\\tSSID: home-net",
    "BSS 99:99:99:99:99:99(on wlan0)", "\\tfreq: 2412.0", "\\tsignal: -30.00 dBm", "\\tSSID: neighbour",
]) + "\\n"

STUBS = {
    "ip": 'case "$*" in *"route show default"*) echo "default via 192.168.50.1 dev wlan0";; esac',
    "ping": (
        'r=$(head -n1 "$STUB/ping_seq"); tail -n +2 "$STUB/ping_seq" > "$STUB/p" && mv "$STUB/p" "$STUB/ping_seq"\n'
        '[ -z "$r" ] && { kill "$PPID"; exit 1; }\n'
        'exit "$r"'
    ),
    "iw": (
        'case "$*" in\n'
        f'  *scan*) [ -f "$STUB/noscan" ] || printf "{SCAN}";;\n'
        f'  *) printf "Connected to {NODE} (on wlan0)\\n\\tfreq: 5500.0\\n\\tsignal: -59 dBm\\n";;\n'
        'esac'
    ),
    "vcgencmd": 'echo "throttled=0x0"',
    "lsmod": 'printf "brcmfmac_cyw 12288 0\\nbrcmfmac 376832 1 brcmfmac_cyw\\n"',
    "journalctl": 'echo "journal-line"',
    "sleep": "exit 0",
    "sync": "exit 0",
    # timeout -k 2 <secs> cmd...: pass straight through.
    "timeout": '[ "$1" = -k ] && shift 2; shift; exec "$@"',
    "nmcli": (
        'case "$*" in\n'
        '  *"GENERAL.CONNECTION device show wlan0"*) echo "home";;\n'
        '  *"802-11-wireless.bssid connection show"*) cat "$STUB/lock";;\n'
        '  *"802-11-wireless.ssid connection show"*) echo "home-net";;\n'
        '  *"-t -f TYPE,NAME connection show"*) printf "loopback:lo\\n802-11-wireless:home\\n";;\n'
        '  *"connection modify home 802-11-wireless.bssid"*) echo "${@: -1}" > "$STUB/lock"; echo "nmcli $*" >> "$STUB/calls";;\n'
        '  "-t -f"*|"radio wifi") ;;\n'
        '  *) echo "nmcli $*" >> "$STUB/calls"; [ -f "$STUB/hang" ] && case "$*" in *off*) exec >/dev/null 2>&1; echo $$ >> "$STUB/hung"; exec /bin/sleep 20;; esac;;\n'
        'esac'
    ),
    "modprobe": (
        'echo "modprobe $*" >> "$STUB/calls"\n'
        '[ -f "$STUB/hang" ] && { exec >/dev/null 2>&1; echo $$ >> "$STUB/hung"; exec /bin/sleep 20; }\n'
        'exit 0'
    ),
    "systemctl": 'echo "systemctl $*" >> "$STUB/calls"',
}


def run(tmp_path, pings, lock="", hang=False, repick=False, noscan=False, **env):
    stub = tmp_path / "bin"
    stub.mkdir()
    for name, body in STUBS.items():
        p = stub / name
        p.write_text(f"#!/bin/bash\n{body}\n")
        p.chmod(0o755)
    (stub / "ping_seq").write_text("".join(f"{r}\n" for r in pings))
    (stub / "calls").write_text("")
    (stub / "lock").write_text(lock + "\n" if lock else "")
    if hang:
        (stub / "hang").write_text("")
    if noscan:
        (stub / "noscan").write_text("")
    request = tmp_path / "wifi-repick.request"
    if repick:
        request.write_text("")
    logdir = tmp_path / "incidents"
    full_env = {
        **os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}",
        "STUB": str(stub),
        "NETWATCH_INTERVAL": "0",
        "NETWATCH_INCIDENT_CHECKS": "3",
        "NETWATCH_RADIO_CHECKS": "4",
        "NETWATCH_UNLOCK_CHECKS": "6",
        "NETWATCH_RELOAD_CHECKS": "8",
        "NETWATCH_REBOOT_CHECKS": "10",
        "NETWATCH_MIN_UPTIME": "0",
        "NETWATCH_DIR": str(logdir),
        "NETWATCH_REPICK": str(request),
        **env,
    }
    # Output goes to a file: hung background stubs must not hold a pipe open.
    with open(tmp_path / "out", "w") as out:
        subprocess.run(["bash", str(SCRIPT)], env=full_env, stdout=out,
                       stderr=subprocess.STDOUT, timeout=15)
    hung = stub / "hung"
    if hung.exists():
        for pid in hung.read_text().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    calls = (stub / "calls").read_text().splitlines()
    incidents = sorted(logdir.glob("incident-*.log")) if logdir.exists() else []
    return {
        "out": (tmp_path / "out").read_text(),
        "raw_calls": calls,
        "lock": (stub / "lock").read_text().strip(),
        "incidents": incidents,
        "request_left": request.exists(),
    }


def actions(r):
    """Ladder actions only: drop the startup 'radio on' and auto-lock writes."""
    calls = r["raw_calls"]
    if calls[:1] == ["nmcli radio wifi on"]:
        calls = calls[1:]
    return [c for c in calls if "802-11-wireless.bssid " + NODE not in c]


def test_never_reboots_before_the_gateway_has_answered_once(tmp_path):
    r = run(tmp_path, [1] * 20)
    assert "systemctl reboot" not in r["raw_calls"]
    assert "nmcli radio wifi off" in r["raw_calls"]   # the gentle steps still run
    # A boot onto a dead locked node still re-picks a live one.
    assert f"nmcli connection modify home 802-11-wireless.bssid {STRONG}" in r["raw_calls"]
    assert len(r["incidents"]) == 1 and "DOWN" in r["incidents"][0].read_text()


def test_short_blips_write_no_incident(tmp_path):
    r = run(tmp_path, [0, 1, 1, 0, 1, 0])
    assert r["incidents"] == []
    assert actions(r) == []


def test_climbs_radio_unlock_driver_then_one_reboot(tmp_path):
    r = run(tmp_path, [0] + [1] * 20)
    a = actions(r)
    assert a[:2] == ["nmcli radio wifi off", "nmcli radio wifi on"]
    assert a[2] == f"nmcli connection modify home 802-11-wireless.bssid {STRONG}"
    assert a[3] == "nmcli connection up home"
    assert a[4] == "modprobe -r brcmfmac_cyw brcmfmac"
    assert a[5] == "modprobe -a brcmfmac brcmfmac_cyw"
    assert a.count("systemctl reboot") == 1


def test_reboot_still_happens_when_radio_and_driver_steps_hang(tmp_path):
    r = run(tmp_path, [0] + [1] * 20, hang=True)
    assert "systemctl reboot" in r["raw_calls"]


def test_no_reboot_before_min_uptime(tmp_path):
    r = run(tmp_path, [0] + [1] * 20, NETWATCH_MIN_UPTIME="999999999")
    assert "systemctl reboot" not in r["raw_calls"]
    assert "nmcli radio wifi off" in r["raw_calls"]


def test_incident_file_records_the_step_that_preceded_recovery(tmp_path):
    r = run(tmp_path, [0, 1, 1, 1, 1, 0])
    for _ in range(20):  # the snapshot and journal dump are written in the background
        text = r["incidents"][0].read_text()
        if "journal-line" in text:
            break
        time.sleep(0.1)
    assert "DOWN" in text and "journal-line" in text
    assert "UP after 4 failed checks" in text and "the radio toggle" in text
    assert "<5>UP after 4" in r["out"]


def test_locks_an_unlocked_profile_to_the_current_node(tmp_path):
    r = run(tmp_path, [0, 0])
    assert r["lock"] == NODE
    assert f"locked WiFi profile 'home' to node {NODE}" in r["out"]


def test_keeps_an_existing_lock(tmp_path):
    r = run(tmp_path, [0, 0], lock="11:22:33:44:55:66")
    assert r["lock"] == "11:22:33:44:55:66"
    assert "locked WiFi profile" not in r["out"]


def test_repick_locks_the_strongest_node_on_the_same_ssid(tmp_path):
    r = run(tmp_path, [0, 0], lock="11:22:33:44:55:66", repick=True)
    a = actions(r)
    assert a[:2] == [f"nmcli connection modify home 802-11-wireless.bssid {STRONG}",
                     "nmcli connection up home"]
    assert r["lock"] == STRONG
    assert not r["request_left"]


def test_repick_without_scan_results_unlocks_then_relocks_where_it_lands(tmp_path):
    r = run(tmp_path, [0, 0], lock="11:22:33:44:55:66", repick=True, noscan=True)
    assert actions(r)[:2] == ["nmcli connection modify home 802-11-wireless.bssid ",
                              "nmcli connection up home"]
    assert r["lock"] == NODE
