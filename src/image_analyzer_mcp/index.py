import os
import cv2
import numpy as np
try:
    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    # mcp < 2.0
    from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Image Sprite Analyzer")

def load_image_safely(image_path: str):
    """Windowsの日本語パスにも対応した安全な画像読み込み"""
    if not os.path.isfile(image_path):
        return None
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

def save_image_safely(image_path: str, img: np.ndarray) -> bool:
    """Windowsの日本語パスにも対応した安全な画像保存"""
    try:
        ext = os.path.splitext(image_path)[1]
        if not ext:
            ext = ".png"
            image_path += ext
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(image_path)
            return True
        return False
    except Exception:
        return False

@mcp.tool()
def get_image_info(image_path: str) -> dict:
    """
    画像の全体サイズ（width, height, channels）やフォーマット情報を素早く取得します。
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    return {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "has_alpha": bool(channels == 4),
        "file_size_bytes": os.path.getsize(image_path),
    }

@mcp.tool()
def find_sub_image_boxes(
    image_path: str,
    min_size: int = 16,
    padding: int = 2,
    bg_mode: str = "auto"
) -> dict:
    """
    画像内の独立したサブ画像（スプライト、コラージュ要素）のバウンディングボックス(x, y, width, height)を検出します。

    Parameters:
        image_path: 解析する画像ファイルのパス
        min_size: 検出対象とする最小幅・最小高さ(px)
        padding: 離れたパーツ（頭と胴体など）を同一オブジェクトとしてまとめる膨張サイズ(px)
        bg_mode: 背景判定モード ('auto', 'transparent', 'white', 'black')
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    if bg_mode == "transparent" and channels != 4:
        raise ValueError(
            f"bg_mode='transparent' が指定されましたが、画像にアルファチャンネルがありません (channels={channels})。"
        )

    # マスクの作成
    if channels == 4 and (bg_mode in ("auto", "transparent")):
        # アルファチャンネルが0より大きい部分を物体とする
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        if channels >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if bg_mode == "auto":
            # 四隅のピクセル輝度の平均から白背景か黒背景かを判定
            corners = [int(gray[0, 0]), int(gray[0, -1]), int(gray[-1, 0]), int(gray[-1, -1])]
            is_light_bg = (sum(corners) / 4) > 127
        else:
            is_light_bg = (bg_mode == "white")

        if is_light_bg:
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # 近接パーツを結合するための膨張処理（連結成分の判定にのみ使用し、
    # バウンディングボックス自体は元のマスクから測り直す）
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (padding * 2 + 1, padding * 2 + 1))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        dilated_mask = mask

    num_labels, labels = cv2.connectedComponents(dilated_mask)

    boxes = []
    for label in range(1, num_labels):
        ys, xs = np.where((labels == label) & (mask > 0))
        if ys.size == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
        if bw >= min_size and bh >= min_size:
            boxes.append({
                "x": x,
                "y": y,
                "width": bw,
                "height": bh
            })

    # 左上から順にソート（各行の基準位置・高さに合わせて隣接する行を逐次クラスタリング）
    boxes.sort(key=lambda b: (b["y"], b["x"]))

    rows = []
    current_row = []
    row_top = None
    row_height = None
    for b in boxes:
        if current_row and b["y"] > row_top + row_height * 0.7:
            rows.append(current_row)
            current_row = [b]
            row_top = b["y"]
            row_height = b["height"]
        else:
            current_row.append(b)
            row_top = min(row_top, b["y"]) if row_top is not None else b["y"]
            row_height = min(row_height, b["height"]) if row_height is not None else b["height"]
    if current_row:
        rows.append(current_row)

    boxes = []
    for row in rows:
        row.sort(key=lambda b: b["x"])
        boxes.extend(row)

    for idx, b in enumerate(boxes):
        b["index"] = idx

    return {
        "image_size": {"width": int(w), "height": int(h)},
        "count": len(boxes),
        "boxes": boxes
    }

@mcp.tool()
def get_grid_boxes(
    image_path: str,
    rows: int = 0,
    cols: int = 0,
    tile_width: int = 0,
    tile_height: int = 0,
    margin_x: int = 0,
    margin_y: int = 0,
    spacing_x: int = 0,
    spacing_y: int = 0
) -> dict:
    """
    規則的に並んでいる等間隔スプライトシートを格子状（グリッド）に分割したバウンディングボックスを算出します。
    rows/cols 指定、または tile_width/tile_height 指定のどちらでも利用可能です。
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]

    # rows / cols または tile_width / tile_height の算出
    if rows > 0 and cols > 0:
        avail_w = w - margin_x * 2 - spacing_x * (cols - 1)
        avail_h = h - margin_y * 2 - spacing_y * (rows - 1)
        tw = avail_w // cols
        th = avail_h // rows
    elif tile_width > 0 and tile_height > 0:
        tw = tile_width
        th = tile_height
        cols = (w - margin_x * 2 + spacing_x) // (tw + spacing_x)
        rows = (h - margin_y * 2 + spacing_y) // (th + spacing_y)
    else:
        raise ValueError("rows と cols、または tile_width と tile_height のいずれかの組み合わせを指定してください。")

    if tw <= 0 or th <= 0 or rows <= 0 or cols <= 0:
        raise ValueError("計算されたタイルのサイズまたは行・列数が不正です。マージンや間隔の設定を確認してください。")

    boxes = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (tw + spacing_x)
            y = margin_y + r * (th + spacing_y)
            boxes.append({
                "index": idx,
                "row": r,
                "col": c,
                "x": int(x),
                "y": int(y),
                "width": int(tw),
                "height": int(th)
            })
            idx += 1

    return {
        "image_size": {"width": int(w), "height": int(h)},
        "grid": {
            "rows": int(rows),
            "cols": int(cols),
            "tile_width": int(tw),
            "tile_height": int(th)
        },
        "count": len(boxes),
        "boxes": boxes
    }

@mcp.tool()
def crop_and_save_sub_images(
    image_path: str,
    boxes: list[dict],
    output_dir: str,
    prefix: str = "sub_image"
) -> dict:
    """
    検出・指定したバウンディングボックスのリストを元に、元画像から個別の画像を切り出して指定ディレクトリに保存します。
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for i, box in enumerate(boxes):
        bx = max(0, int(box.get("x", 0)))
        by = max(0, int(box.get("y", 0)))
        bw = int(box.get("width", 0))
        bh = int(box.get("height", 0))

        # 範囲クリッピング
        bx2 = min(w, bx + bw)
        by2 = min(h, by + bh)

        if bx2 <= bx or by2 <= by:
            continue

        cropped = img[by:by2, bx:bx2]
        filename = f"{prefix}_{i:03d}.png"
        out_path = os.path.join(output_dir, filename)

        if save_image_safely(out_path, cropped):
            saved_files.append({
                "index": i,
                "path": out_path,
                "width": int(bx2 - bx),
                "height": int(by2 - by)
            })

    return {
        "saved_count": len(saved_files),
        "output_directory": os.path.abspath(output_dir),
        "files": saved_files
    }

def main():
    mcp.run()

if __name__ == "__main__":
    main()
