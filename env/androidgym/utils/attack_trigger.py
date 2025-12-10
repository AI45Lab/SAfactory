import time
from typing import Callable
from env.androidgym.utils.foreground_hook import send_broadcast_for_overlay, send_emulator_sms

def build_overlay_trigger(
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
) -> Callable[[], None]:
    def _trigger() -> None:
        print("[hook] App detected in foreground, sending broadcast...")
        send_broadcast_for_overlay(
            adb_path=adb_path,
            component=component,
            action=action,
            title=title,
            content=content,
            cancel=cancel,
            confirm=confirm,
            weburl=weburl,
            is_urgent=is_urgent,
        )
        time.sleep(1)
    return _trigger

def build_popup_trigger(*, adb_path: str, phone: str, content: str) -> Callable[[], None]:
    def _trigger() -> None:
        print("[hook] App detected in foreground, injecting inbound SMS via emulator console...")
        send_emulator_sms(adb_path=adb_path, phone=phone, content=content)
        time.sleep(0.5)
    return _trigger
