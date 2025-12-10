import subprocess, threading, time, re
from typing import List, Optional

COMMON_APPS = {
    "+ contact": "com.google.android.contacts",
    "AliExpress": "com.alibaba.aliexpresshd",
    "Book App": "com.booking",
    "Booking": "com.booking",
    "Calendar": "com.google.android.calendar",
    "Camera": "com.android.camera2",
    "Chrome": "com.android.chrome",
    "Files": "com.google.android.documentsui",
    "Gallery": "com.google.android.apps.photosgo",
    "Gmail": "com.google.android.gm",
    "Go Forex": "com.tiim.goforexx24",
    "Messages": "com.google.android.apps.messaging",
    "Nextcloud": "com.nextcloud.client",
    "Nextcloud Dev": "com.nextcloud.client",
    "PhotoNote": "com.chartreux.photo_note",
    "Photos": "com.google.android.apps.photosgo",
    "Settings": "com.android.settings",
    "System UI": None,
    "messages": "com.google.android.apps.messaging",

}

class ForegroundAppHook:
    def __init__(
        self,
        adb_path: str,
        apps: Optional[List[str]] = None,   #
        poll_interval: float = 0.7,
        stable_checks: int = 3,
        debounce_seconds: int = 30
    ):
        self.adb_path = adb_path
        self.apps = list(apps or [])        
        self.poll_interval = poll_interval
        self.stable_checks = stable_checks
        self.debounce_seconds = debounce_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_trigger_ts = 0.0
        self._triggered = False
        self.on_trigger = None

    def start(self, on_trigger):
        if (self._thread and self._thread.is_alive()) or self._triggered:
            return
        self.on_trigger = on_trigger
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ForegroundAppHook")
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=3)

    def _adb(self, *args: str) -> str:
        out = subprocess.run([self.adb_path, *args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return out.stdout or ""

    def _get_top_pkg(self) -> Optional[str]:
        candidates: list[tuple[list[str], list[re.Pattern]]] = [
            (["shell", "dumpsys", "activity", "activities"],
             [re.compile(r"topResumedActivity.+\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+"),
              re.compile(r"mResumedActivity:.+\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+")]),
            (["shell", "dumpsys", "activity", "top"],
             [re.compile(r"ACTIVITY\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+")]),
            (["shell", "dumpsys", "window", "windows"],
             [re.compile(r"mCurrentFocus=Window\{.+\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+"),
              re.compile(r"Current\s+focus=Window\{.+\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+")]),
            (["shell", "dumpsys", "window"],
             [re.compile(r"mFocusedApp=AppWindowToken\{.+\s([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+")]),
        ]
        for cmd, patterns in candidates:
            out = self._adb(*cmd)
            if not out:
                continue
            for p in patterns:
                m = p.search(out)
                if m:
                    return m.group(1)
        return None

    def _resolve_pkg(self, app: str) -> Optional[str]:
        if app == "System UI":
            return None
        if app in COMMON_APPS:
            return COMMON_APPS[app]

        if "." in app:
            return app
        return None

    def _target_pkgs(self) -> List[str]:
        pkgs: List[str] = []
        for app in self.apps:
  
            if isinstance(app, list):
                for a in app:
                    p = self._resolve_pkg(str(a))
                    if p: pkgs.append(p)
            else:
                p = self._resolve_pkg(str(app))
                if p: pkgs.append(p)

        return list(dict.fromkeys(pkgs))

    def _loop(self):
        target_pkgs = self._target_pkgs()
        if not target_pkgs:
            print(f"[ForegroundAppHook] warning: no valid target packages from apps={self.apps!r}")
            self._stop.set()
            return

        stable_pkg = None
        stable_count = 0

        while not self._stop.is_set() and not self._triggered:
            try:
                pkg = self._get_top_pkg()
                if pkg == stable_pkg and pkg is not None:
                    stable_count += 1
                else:
                    stable_pkg = pkg
                    stable_count = 1

                now = time.time()
                ready = (
                    pkg in target_pkgs                      
                    and stable_count >= self.stable_checks
                    and (now - self._last_trigger_ts) >= self.debounce_seconds
                )
                if ready:
                    try:
                        if self.on_trigger:
                            self.on_trigger()
                        self._triggered = True
                    finally:
                        self._last_trigger_ts = now
                        self._stop.set()
                        break
            except Exception as e:
                print(f"[ForegroundAppHook] error: {e}")
            time.sleep(self.poll_interval)

import subprocess

def _dq(s: str) -> str:
    """把字符串安全地放进 POSIX 双引号中。"""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{s}"'

def _build_remote_cmd(
    *, component: str, action: str, title: str, content: str, cancel: str, confirm: str,
    weburl: str, is_urgent: bool
) -> str:

    return (
        "am broadcast "
        f"-n {component} "
        f"-a {action} "
        f"--es title {_dq(title)} "
        f"--es content {_dq(content)} "
        f"--es cancel {_dq(cancel)} "
        f"--es confirm {_dq(confirm)} "
        f"--es weburl {_dq(weburl)} "
        f"--ez isUrgent {'true' if is_urgent else 'false'} "
        f"--receiver-foreground"
    )

def send_broadcast_for_overlay(
    *,
    adb_path: str,
    component: str,
    action: str,
    title: str,
    content: str,
    cancel: str,
    confirm: str,
    weburl: str,
    is_urgent: bool = True,
    timeout: int = 10,
):

    remote_cmd = _build_remote_cmd(
        component=component, action=action, title=title, content=content,
        cancel=cancel, confirm=confirm, weburl=weburl, is_urgent=is_urgent
    )

    return subprocess.run(
        [adb_path, "shell", remote_cmd],
        capture_output=True, text=True, timeout=timeout, check=False
    )


def pick_emulator_serial(adb_path: str) -> str:
    out = subprocess.run([adb_path, "devices"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("emulator-") and "\tdevice" in line:
            return line.split()[0]
    raise RuntimeError(f"No running emulator found.\nADB devices:\n{out}")

def send_emulator_sms(
    *,
    adb_path: str,
    phone: str,
    content: str,
    serial: str | None = None,
    timeout: int = 10,
):

    if not serial:
        serial = pick_emulator_serial(adb_path)

    args = [adb_path, "-s", serial, "emu", "sms", "send", phone, content]
    cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


    if cp.stdout.strip():
        print("[emu sms] stdout:", cp.stdout.strip())
    if cp.stderr and cp.stderr.strip():
        print("[emu sms] stderr:", cp.stderr.strip())

    if cp.returncode != 0:
        raise RuntimeError(f"adb emu sms send failed (rc={cp.returncode})")

    return cp

