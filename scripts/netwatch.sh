#!/usr/bin/env bash
# doggy-netwatch: keep the Pi on the network, lock it to one WiFi node, and
# save the evidence when the network drops.
#
# The Pi's onboard WiFi (brcmfmac) can stop working while Linux keeps running,
# which leaves the dashboard and SSH unreachable until a power cycle. The
# journal lives in RAM (reduce-sd-writes.sh), so a power cycle also erases
# every log of what happened.
#
# Node lock: on a mesh network the Pi otherwise switches between nodes with
# similar signal. Whenever the Pi is connected and its WiFi profile has no
# BSSID set, this locks the profile to the node it is on. The dashboard's
# "Re-pick WiFi node" button creates REPICK; this then scans, locks the
# profile to the strongest node broadcasting its SSID, and reconnects.
#
# Every INTERVAL seconds this pings the default gateway. While the gateway
# stays unreachable it escalates, one step per threshold of failed checks:
#   RADIO_CHECKS   toggle the WiFi radio (NetworkManager reconnects)
#   UNLOCK_CHECKS  re-pick the node (the locked one may be gone)
#   RELOAD_CHECKS  reload the WiFi driver
#   REBOOT_CHECKS  reboot
# A reboot also needs the gateway to have answered once this boot and
# MIN_UPTIME seconds of uptime, so a router that ignores ping or stays off for
# hours causes at most one reboot. Every step that talks to the
# WiFi stack runs in the background with a time limit, so a wedged driver
# cannot stop the ladder from reaching the reboot.
#
# After INCIDENT_CHECKS failed checks it writes an incident file to DIR (on
# the SD card, so it survives a reboot) with a snapshot and the last 30
# minutes of the journal, dumps the journal again before a reboot, and records
# which step the network came back after. Nothing is written to disk while the
# network is up. A status line goes to the journal every STATUS_CHECKS checks.
#
# Updates: systemd runs this file straight from the app's scripts/ directory,
# which deploys and self-updates replace. When the file's contents change it
# exits, and systemd (Restart=always) starts the new version. Running an
# app-user-writable file as root adds no privilege the app user lacks: it has
# passwordless sudo. If that sudo is ever removed, install a root-owned copy
# instead.
set -u

INTERVAL=${NETWATCH_INTERVAL:-20}
INCIDENT_CHECKS=${NETWATCH_INCIDENT_CHECKS:-3}   # ~1 min; ignores single lost pings and boot
RADIO_CHECKS=${NETWATCH_RADIO_CHECKS:-9}         # ~3 min at 20 s
UNLOCK_CHECKS=${NETWATCH_UNLOCK_CHECKS:-18}      # ~6 min
RELOAD_CHECKS=${NETWATCH_RELOAD_CHECKS:-27}      # ~9 min
REBOOT_CHECKS=${NETWATCH_REBOOT_CHECKS:-36}      # ~12 min
STATUS_CHECKS=${NETWATCH_STATUS_CHECKS:-30}      # ~10 min
MIN_UPTIME=${NETWATCH_MIN_UPTIME:-3600}
DIR=${NETWATCH_DIR:-/var/log/doggy-netwatch}
KEEP=${NETWATCH_KEEP:-30}
REPICK=${NETWATCH_REPICK:-/home/doggy/doggy/wifi-repick.request}
# A 5 GHz node at this signal or better beats any 2.4 GHz node: 2.4 GHz always
# reads stronger, and on the Pi it shares the radio with the Bluetooth speaker.
MIN_5GHZ_DBM=${NETWATCH_MIN_5GHZ_DBM:--75}

STEP_NAMES=("nothing (came back by itself)" "the radio toggle" "the node re-pick" "the driver reload" "the reboot")

# t <seconds> <cmd...>: run with a time limit.
t() { local s=$1; shift; timeout -k 2 "$s" "$@"; }

uptime_s() { local u=0; [ -r /proc/uptime ] && read -r u _ </proc/uptime; echo "${u%.*}"; }

current_bssid() { t 5 iw dev wlan0 link 2>/dev/null | awk '/^Connected/{print $3; exit}'; }

link() {
  local l
  l=$(t 5 iw dev wlan0 link 2>/dev/null)
  case $l in
    Connected*) awk '/^Connected/{b=$3} /freq:/{f=$2} /signal:/{s=$2} END{printf "bssid=%s freq=%s signal=%s", b, f, s}' <<<"$l" ;;
    *) printf 'wifi=disconnected' ;;
  esac
}

health() { printf '%s %s %s' "$(link)" "$(t 5 vcgencmd get_throttled 2>/dev/null)" "$(t 5 vcgencmd measure_temp 2>/dev/null)"; }

# say <priority> <message>: to the journal, and to the open incident file.
# <2> (crit) makes journald write to its store right away.
say() {
  echo "<$1>$2"
  [ -n "$incident" ] && echo "$(date '+%F %T') $2" >>"$incident"
}

dump_journal() {
  [ -n "$incident" ] || return 0
  { echo "----- journal, last 30 min ($1)"; t 30 journalctl --since "-30min" --no-pager -o short-iso 2>&1; } >>"$incident"
}

open_incident() {
  mkdir -p "$DIR"
  incident="$DIR/incident-$(date +%Y%m%d-%H%M%S).log"
  : >"$incident"
  local f
  for f in $(ls -1t "$DIR"/incident-*.log 2>/dev/null | tail -n +$((KEEP + 1))); do rm -f "$f"; done
  # In the background: these touch the WiFi stack, which may be wedged.
  (
    {
      echo "----- snapshot: $(health)"
      t 5 ip -br addr show 2>&1
      t 5 ip route show 2>&1
      t 10 nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device 2>&1
      echo "wifi radio: $(t 10 nmcli radio wifi 2>&1) profile: ${wifi_con:-unknown} lock: ${lock:-none}"
    } >>"$incident"
    dump_journal "when the outage was declared"
  ) &
}

# The first WiFi profile, known even while disconnected (a boot where the
# locked node is gone must still be able to re-pick).
wifi_profile() {
  t 10 nmcli -e no -t -f TYPE,NAME connection show 2>/dev/null |
    awk '/^802-11-wireless:/{ sub(/^[^:]*:/, ""); print; exit }'
}

# Lock the WiFi profile to the node the Pi is on, when it has no lock yet.
ensure_lock() {
  local con bssid
  con=$(t 10 nmcli -e no -g GENERAL.CONNECTION device show wlan0 2>/dev/null)
  bssid=$(current_bssid)
  lock_checked=1
  [ -n "$con" ] && [ "$con" != "--" ] && [ -n "$bssid" ] || return 0   # not on WiFi
  wifi_con=$con
  lock=$(t 10 nmcli -e no -g 802-11-wireless.bssid connection show "$con" 2>/dev/null)
  if [ -z "$lock" ] && t 10 nmcli connection modify "$con" 802-11-wireless.bssid "$bssid"; then
    lock=$bssid
    say 5 "locked WiFi profile '$con' to node $bssid. $(health)"
  fi
}

# Lock the profile to the strongest node broadcasting its SSID (5 GHz first,
# see MIN_5GHZ_DBM) and reconnect. Left to itself, NetworkManager joined the
# weaker node in the first live test. With no scan result (a wedged driver), clear the lock and let
# NetworkManager choose.
repick_node() {
  [ -n "$wifi_con" ] || return 0
  local ssid best
  ssid=$(t 10 nmcli -e no -g 802-11-wireless.ssid connection show "$wifi_con" 2>/dev/null)
  best=$(t 30 iw dev wlan0 scan 2>/dev/null | awk -v want="$ssid" -v min5="$MIN_5GHZ_DBM" '
    /^BSS /    { bss = substr($2, 1, 17) }
    /freq:/    { freq = $2 + 0 }
    /signal:/  { sig = $2 + 0 }
    /^\tSSID: / {
      if (substr($0, 8) != want) next
      if (freq > 4000 && sig >= min5 && (b5 == "" || sig > s5)) { b5 = bss; s5 = sig }
      if (any == "" || sig > sa) { any = bss; sa = sig }
    }
    END        { print (b5 != "" ? b5 : any) }')
  say 5 "re-pick: locking to ${best:-no node (empty scan), NetworkManager chooses}"
  t 10 nmcli connection modify "$wifi_con" 802-11-wireless.bssid "$best" &&
    t 90 nmcli connection up "$wifi_con" >/dev/null
}

reload_driver() {
  # brcmfmac's helper modules (brcmfmac_cyw/_wcc/_bca) hold it; unload them
  # first, and load everything back even if the unload was partial.
  local helpers
  helpers=$(lsmod | awk '/^brcmfmac_/{print $1}')
  t 60 modprobe -r $helpers brcmfmac
  sleep 3
  t 60 modprobe -a brcmfmac $helpers
}

gw=""; ever_up=0; failed=0; step=0; checks=0; incident=""
self_sum=$(cksum <"$0" 2>/dev/null)
wifi_con=$(wifi_profile); lock=""; lock_checked=0

# A radio toggle cut short (netwatch killed between off and on) would leave
# WiFi off for good, because NetworkManager remembers the radio state.
t 10 nmcli radio wifi on

while :; do
  if [ "$(cksum <"$0" 2>/dev/null)" != "$self_sum" ]; then
    say 5 "netwatch.sh changed on disk; exiting so systemd starts the new version"
    exit 0
  fi
  # A regular file only: anything rm -f can't clear would re-pick forever.
  if [ -f "$REPICK" ] && rm -f "$REPICK"; then
    [ -n "$wifi_con" ] || ensure_lock
    say 5 "re-pick requested (node lock was ${lock:-none})"
    repick_node   # in the foreground: the next check must see the new node
    lock=""; lock_checked=0
  fi

  route=$(t 5 ip -4 route show default 2>/dev/null | awk '{print $3; exit}')
  [ -n "$route" ] && gw=$route
  checks=$((checks + 1))

  if [ -n "$gw" ] && ping -c1 -W3 -w5 "$gw" >/dev/null 2>&1; then
    if [ "$failed" -ge "$INCIDENT_CHECKS" ]; then
      say 5 "UP after $failed failed checks (~$((failed * INTERVAL)) s); last action before recovery: ${STEP_NAMES[$step]}. $(health)"
      lock_checked=0   # the node may have changed (unlock step, driver reload)
    elif [ "$ever_up" -eq 0 ]; then
      say 6 "watching gateway $gw. $(health)"
    fi
    ever_up=1; failed=0; step=0; incident=""
    [ "$lock_checked" -eq 0 ] && ensure_lock
    [ $((checks % STATUS_CHECKS)) -eq 0 ] && say 6 "ok gateway $gw lock ${lock:-none}. $(health)"
  else
    failed=$((failed + 1))
    if [ "$failed" -eq "$INCIDENT_CHECKS" ]; then
      open_incident
      say 2 "DOWN: gateway ${gw:-unknown} has not answered for $failed checks"
    elif [ $((failed % STATUS_CHECKS)) -eq 0 ]; then
      say 4 "still down after $failed checks"
    fi

    # Steps 1-3 also run on a boot where WiFi never came up; the reboot does not.
    if [ "$step" -lt 1 ] && [ "$failed" -ge "$RADIO_CHECKS" ]; then
      step=1
      say 3 "step 1: toggling the WiFi radio"
      ( t 15 nmcli radio wifi off; sleep 3; t 15 nmcli radio wifi on ) &
    elif [ "$step" -lt 2 ] && [ "$failed" -ge "$UNLOCK_CHECKS" ]; then
      step=2
      say 3 "step 2: re-picking the WiFi node (lock was ${lock:-none})"
      repick_node &
      lock=""; lock_checked=0
    elif [ "$step" -lt 3 ] && [ "$failed" -ge "$RELOAD_CHECKS" ]; then
      step=3
      say 3 "step 3: reloading the WiFi driver"
      reload_driver &
    elif [ "$step" -lt 4 ] && [ "$failed" -ge "$REBOOT_CHECKS" ] && [ "$ever_up" -eq 1 ] && [ "$(uptime_s)" -ge "$MIN_UPTIME" ]; then
      step=4
      say 2 "step 4: rebooting"
      dump_journal "before reboot"
      sync
      # A refused clean reboot (inhibitor, D-Bus trouble) falls back to an
      # immediate one; the journal dump and sync above already ran.
      t 30 systemctl reboot || t 30 systemctl reboot --force --force
    fi
  fi
  sleep "$INTERVAL"
done
