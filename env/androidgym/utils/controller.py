import subprocess
from pathlib import Path
from PIL import Image
import shutil

def get_screenshot(adb_path: str, device_serial: str, out_dir: str = "screenshot", base: str = "screenshot") -> str:
    judgement_dir = Path("env/androidgym/judgement")
    judgement_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{base}.png"
    jpg = out / f"{base}.jpg"

    # 先清理本地旧文件
    try:
        if png.exists(): png.unlink()
        if jpg.exists(): jpg.unlink()
    except Exception:
        pass

    # 尝试方案 A：exec-out（最快，避免换行符问题）
    try:
        with open(png, "wb") as f:
            res = subprocess.run(
                [adb_path, "-s", device_serial, "exec-out", "screencap", "-p"],
                stdout=f, stderr=subprocess.PIPE, check=True
            )
        if not png.exists() or png.stat().st_size < 1000:
            raise RuntimeError("exec-out screencap produced empty file")
    except Exception as e:
        # 回退方案 B：存到 /sdcard 再拉回
        subprocess.run([adb_path, "-s", device_serial, "shell", "rm", "-f", "/sdcard/__tmp_screenshot.png"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r1 = subprocess.run([adb_path, "-s", device_serial, "shell", "screencap", "-p", "/sdcard/__tmp_screenshot.png"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r1.returncode != 0:
            raise RuntimeError(f"screencap failed: {r1.stderr.decode(errors='ignore')}")
        r2 = subprocess.run([adb_path, "-s", device_serial, "pull", "/sdcard/__tmp_screenshot.png", str(png)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r2.returncode != 0 or not png.exists():
            raise RuntimeError(f"adb pull failed: {r2.stderr.decode(errors='ignore')}")
        subprocess.run([adb_path, "-s", device_serial, "shell", "rm", "-f", "/sdcard/__tmp_screenshot.png"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 转 JPG
    with Image.open(png) as im:
        im.convert("RGB").save(jpg, "JPEG", quality=95)

    # 清理临时 png
    try:
        png.unlink()
    except Exception:
        pass

    # 按照a,b,c,d...的顺序保存到judgement_dir
    # 获取当前judgement_dir中已有的文件数量
    existing_files = list(judgement_dir.glob("*.jpg"))
    next_letter = chr(ord('a') + len(existing_files))
    
    # 如果超过26个文件，使用双字母命名(aa, ab, ac...)
    if len(existing_files) >= 26:
        first_letter = chr(ord('a') + (len(existing_files) // 26) - 1)
        second_letter = chr(ord('a') + (len(existing_files) % 26))
        next_filename = f"{first_letter}{second_letter}.jpg"
    else:
        next_filename = f"{next_letter}.jpg"
    
    # 复制文件到judgement_dir
    judgement_file = judgement_dir / next_filename
    shutil.copy2(jpg, judgement_file)
    
    return str(jpg)

def tap(adb_path, device_serial, x, y):
    command = adb_path + f" -s {device_serial} shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, device_serial, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" -s {device_serial} shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" -s {device_serial} shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" -s {device_serial} shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" -s {device_serial} shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" -s {device_serial} shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)


def slide(adb_path, device_serial, x1, y1, x2, y2):
    command = adb_path + f" -s {device_serial} shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def home(adb_path, device_serial):
    command = adb_path + f" -s {device_serial} shell input keyevent 3"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path, device_serial):
    command = adb_path + f" -s {device_serial} shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)


# def home(adb_path):
#     command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
#     subprocess.run(command, capture_output=True, text=True, shell=True)

def take_screenshot(adb_path, device_serial):
    command = adb_path + ' -s {device_serial} shell screencap -p /storage/emulated/0/Pictures/1.png'
    subprocess.run(command, capture_output=True, text=True, shell=True)