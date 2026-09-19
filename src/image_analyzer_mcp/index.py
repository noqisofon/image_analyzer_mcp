# image_analyzer_mcp.py
import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Image Sprite Analyzer")

@mcp.tool()
def find_sub_image_boxes(image_path: str, min_size: int = 16) -> list[dict]:
    """画像内の独立したサブ画像のバウンディングボックス(x, y, w, h)を検出します。"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像が読み込めません")

    # 透過PNGならアルファチャンネル、それ以外は二値化して輪郭抽出
    if img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_size and h >= min_size:
            results.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

    # 左上から順にソート
    results.sort(key=lambda b: (b["y"] // 20, b["x"]))
    return results

if __name__ == "__main__":
    mcp.run()
