# board_selector.py
import cv2

# 显示的最大尺寸（适配屏幕，不超过这个范围）
MAX_DISPLAY_W = 600
MAX_DISPLAY_H = 800


def _fit_to_screen(img):
    """缩放图像以适应屏幕，返回 (缩放后图像, 缩放比例)"""
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale >= 1.0:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def select_board_region(image):
    """框选棋盘区域 —— 图像自动缩放适配屏幕，坐标转回原始像素"""
    display_img, scale = _fit_to_screen(image)

    rect = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    drawing = {"flag": False, "ix": -1, "iy": -1}

    # 克隆一份用于重置
    base_display = display_img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal rect, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["flag"] = True
            drawing["ix"] = x
            drawing["iy"] = y

        elif event == cv2.EVENT_MOUSEMOVE and drawing["flag"]:
            temp = base_display.copy()
            cv2.rectangle(temp, (drawing["ix"], drawing["iy"]), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Board", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing["flag"] = False
            x1, x2 = sorted([drawing["ix"], x])
            y1, y2 = sorted([drawing["iy"], y])
            rect["x1"], rect["y1"], rect["x2"], rect["y2"] = x1, y1, x2, y2
            temp = base_display.copy()
            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select Board", temp)

    cv2.namedWindow("Select Board", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Board", mouse_callback)

    print(f"  原始尺寸: {image.shape[1]}×{image.shape[0]} → 显示尺寸: {display_img.shape[1]}×{display_img.shape[0]}")
    print("  请用鼠标框选棋盘区域，Enter 确认，ESC 取消")

    while True:
        cv2.imshow("Select Board", base_display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s") or key == 13:  # s 或 Enter 确认
            break
        elif key == 27:  # ESC 取消
            cv2.destroyWindow("Select Board")
            return None

    cv2.destroyWindow("Select Board")

    if rect["x1"] == rect["x2"] or rect["y1"] == rect["y2"]:
        print("未检测到有效框选区域")
        return None

    # 坐标从显示尺寸转回原始尺寸
    inv_scale = 1.0 / scale
    x1 = int(rect["x1"] * inv_scale)
    y1 = int(rect["y1"] * inv_scale)
    x2 = int(rect["x2"] * inv_scale)
    y2 = int(rect["y2"] * inv_scale)

    print(f"  选中区域: ({x1},{y1}) → ({x2},{y2})")
    return (x1, y1, x2, y2)
