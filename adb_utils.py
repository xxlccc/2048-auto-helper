# adb_utils.py
import subprocess
import cv2
import numpy as np

def adb_screenshot(device_id=None):
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += ["exec-out", "screencap", "-p"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    data = result.stdout
    if not data:
        return None

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image
