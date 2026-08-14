"""The templated payloads the onboarding door serves: the door HTML page
and the Apple .mobileconfig profile that installs the home CA."""
from __future__ import annotations

import base64
import ssl
import uuid
from pathlib import Path

# Stable namespace so a re-downloaded profile carries the same PayloadUUID and
# the device treats it as the same profile rather than a second copy.
_PROFILE_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "watchdoggy.ca.onboarding")


def _ca_der_b64(ca_path: Path) -> str:
    """The home CA as base64-encoded DER (what the .mobileconfig payload wants)."""
    return base64.b64encode(ssl.PEM_cert_to_DER_cert(ca_path.read_text())).decode()


def mobileconfig(ca_path: Path) -> str:
    der_b64 = _ca_der_b64(ca_path)
    cert_uuid = uuid.uuid5(_PROFILE_NS, "cert:" + der_b64)
    profile_uuid = uuid.uuid5(_PROFILE_NS, "profile:" + der_b64)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "  <key>PayloadContent</key>\n"
        "  <array>\n"
        "    <dict>\n"
        "      <key>PayloadCertificateFileName</key>\n"
        "      <string>watchdoggy-ca.pem</string>\n"
        "      <key>PayloadContent</key>\n"
        f"      <data>{der_b64}</data>\n"
        "      <key>PayloadDescription</key>\n"
        "      <string>Adds the watchdoggy home CA so this device trusts the dashboard.</string>\n"
        "      <key>PayloadDisplayName</key>\n"
        "      <string>watchdoggy home CA</string>\n"
        "      <key>PayloadIdentifier</key>\n"
        f"      <string>com.watchdoggy.ca.{cert_uuid}</string>\n"
        "      <key>PayloadType</key>\n"
        "      <string>com.apple.security.root</string>\n"
        "      <key>PayloadUUID</key>\n"
        f"      <string>{cert_uuid}</string>\n"
        "      <key>PayloadVersion</key>\n"
        "      <integer>1</integer>\n"
        "    </dict>\n"
        "  </array>\n"
        "  <key>PayloadDisplayName</key>\n"
        "  <string>watchdoggy home CA</string>\n"
        "  <key>PayloadIdentifier</key>\n"
        f"  <string>com.watchdoggy.ca.profile.{profile_uuid}</string>\n"
        "  <key>PayloadType</key>\n"
        "  <string>Configuration</string>\n"
        "  <key>PayloadUUID</key>\n"
        f"  <string>{profile_uuid}</string>\n"
        "  <key>PayloadVersion</key>\n"
        "  <integer>1</integer>\n"
        "</dict>\n"
        "</plist>\n"
    )



def door_page(ssl_port: int) -> str:
    return _PAGE.replace("__SSL_PORT__", str(ssl_port))


# Server-templated with the https port. Same night-watch language as the
# dashboard (dark, amber, serif wordmark; plain-language copy, no emoji).
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>watchdoggy &middot; secure this device</title>
<style>
  :root{
    --night:#171310; --walnut:#211b15; --seam:#372e23;
    --linen:#efe7d8; --stone:#9e9179; --lamp:#e2a23b; --lamp-soft:#f0be6b;
    --serif:"Iowan Old Style","Palatino","Book Antiqua",Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--night);color:var(--linen);
       font-family:var(--sans);line-height:1.55;
       display:flex;justify-content:center;padding:2.2rem 1.1rem}
  :focus-visible{outline:2px solid var(--lamp);outline-offset:2px;border-radius:4px}
  main{width:100%;max-width:32rem}
  .mark{font-family:var(--serif);font-style:italic;font-weight:500;
        font-size:1.7rem;color:var(--linen)}
  .sub{display:block;font-family:var(--mono);font-size:.62rem;letter-spacing:.24em;
       text-transform:uppercase;color:var(--stone);margin-top:.3rem}
  h1{font-size:1.35rem;font-weight:600;margin:1.6rem 0 .5rem}
  p{color:var(--linen);margin:.55rem 0}
  .muted{color:var(--stone);font-size:.92rem}
  .card{background:var(--walnut);border:1px solid var(--seam);border-radius:10px;
        padding:1.15rem 1.25rem;margin-top:1.4rem}
  .status{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;
          text-transform:uppercase;color:var(--lamp)}
  ol{margin:.6rem 0 0;padding-left:1.2rem}
  li{margin:.35rem 0;color:var(--linen)}
  .platform{font-family:var(--mono);font-size:.62rem;letter-spacing:.14em;
            text-transform:uppercase;color:var(--stone);margin-bottom:.35rem}
  a.download,button{font-family:var(--sans);font-size:.95rem;font-weight:600;
        display:inline-block;padding:.6rem 1.1rem;border-radius:8px;cursor:pointer;
        text-decoration:none;background:var(--lamp);color:#1a130a;border:1px solid var(--lamp)}
  a.download:hover,button:hover{background:var(--lamp-soft);border-color:var(--lamp-soft)}
  .row{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;margin-top:1rem}
  .hidden{display:none}
  code{font-family:var(--mono);color:var(--lamp-soft);font-size:.85rem}
</style>
</head>
<body>
<main>
  <span class="mark">watchdoggy</span>
  <span class="sub">night watch</span>

  <div id="checking" class="card">
    <span class="status">Checking this device</span>
    <p class="muted">Making sure the connection to the dashboard is secure. One moment.</p>
  </div>

  <section id="setup" class="hidden">
    <h1>Secure this device</h1>
    <p>This device has not trusted the home certificate yet, so the dashboard
       will not open securely. Install it once and the padlock stays normal
       from then on. Nothing here talks to the internet.</p>

    <div class="card">
      <div class="platform" id="platform">Step 1</div>
      <p>Download the home certificate, then trust it:</p>
      <a class="download" id="download" href="/ca.pem">Download certificate</a>
      <ol id="steps"></ol>
    </div>

    <div class="row">
      <button id="again" type="button">Check again</button>
      <span class="muted">or just finish the steps and this page notices on its own.</span>
    </div>
    <p class="muted" style="margin-top:1.1rem">Direct address once trusted:
       <code id="direct"></code></p>
  </section>
</main>

<script>
  var SSL_PORT = __SSL_PORT__;
  var httpsBase = "https://" + location.hostname + ":" + SSL_PORT + "/";
  var checking = document.getElementById("checking");
  var setup = document.getElementById("setup");
  document.getElementById("direct").textContent = httpsBase;

  var ua = navigator.userAgent;
  var isApple = /iPhone|iPad|Macintosh/.test(ua);
  var isAndroid = /Android/.test(ua);
  var dl = document.getElementById("download");
  dl.setAttribute("href", isApple ? "/ca.mobileconfig" : "/ca.pem");

  var stepsByPlatform = {
    apple: ["Open the downloaded profile and let it install.",
            "Open Settings, tap the profile at the top, and tap Install.",
            "On iPhone or iPad, also open Settings, General, About, Certificate Trust Settings and turn it on."],
    android: ["Open Settings, Security, Install a certificate, CA certificate.",
              "Pick the file you just downloaded and confirm."],
    other: ["Open the downloaded file.",
            "When asked, trust it for identifying websites (set trust to Always)."]
  };
  var key = isApple ? "apple" : (isAndroid ? "android" : "other");
  document.getElementById("platform").textContent =
    key === "apple" ? "iPhone, iPad, or Mac" : (key === "android" ? "Android" : "This device");
  var ol = document.getElementById("steps");
  stepsByPlatform[key].forEach(function(t){
    var li = document.createElement("li"); li.textContent = t; ol.appendChild(li);
  });

  var done = false;
  function showSetup(){ if(done) return; checking.classList.add("hidden"); setup.classList.remove("hidden"); }
  function probe(){
    // An http page fetching this https URL fails at the TLS layer until the
    // CA is trusted; success means the padlock is clean, so send them in.
    fetch(httpsBase + "ping", {mode: "cors"})
      .then(function(){ done = true; location.replace(httpsBase); })
      .catch(function(){ showSetup(); });
  }
  document.getElementById("again").addEventListener("click", probe);
  probe();
  setInterval(probe, 5000);
</script>
</body>
</html>
"""
