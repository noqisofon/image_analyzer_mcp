import os
import cv2
import numpy as np
from mcp.server.mcpserver import Image, MCPServer

mcp = MCPServer("Image Sprite Analyzer")

def load_image_safely(image_path: str):
    """Windowsの日本語パスにも対応した安全な画像読み込み。16bit画像は8bitに正規化します。"""
    if not os.path.isfile(image_path):
        return None
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        return img
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
    bg_mode: str = "auto",
    bg_color: list[int] | None = None,
    tolerance: int = 20
) -> dict:
    """
    画像内の独立したサブ画像（スプライト、コラージュ要素）のバウンディングボックス(x, y, width, height)を検出します。

    Parameters:
        image_path: 解析する画像ファイルのパス
        min_size: 検出対象とする最小幅・最小高さ(px)
        padding: 離れたパーツ（頭と胴体など）を同一オブジェクトとしてまとめる膨張サイズ(px)
        bg_mode: 背景判定モード ('auto', 'transparent', 'white', 'black', 'color')
        bg_color: 明示的な背景色指定 [R, G, B] (0-255)。bg_mode='color' または指定時に使用
        tolerance: 単色背景との色の許容差(0-255)
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
    if bg_color is not None or bg_mode == "color":
        if bg_color is None:
            raise ValueError("bg_mode='color' を指定する場合は bg_color=[R, G, B] を指定してください。")
        if len(bg_color) != 3:
            raise ValueError("bg_color は [R, G, B] の3要素で指定してください。")
        target_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.int16)
        if channels >= 3:
            diff = np.abs(img[:, :, :3].astype(np.int16) - target_bgr)
            dist = np.max(diff, axis=2)
            mask = np.where(dist > tolerance, 255, 0).astype(np.uint8)
        else:
            gray_target = int(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
            diff = np.abs(img.astype(np.int16) - gray_target)
            mask = np.where(diff > tolerance, 255, 0).astype(np.uint8)
    elif bg_mode == "transparent":
        # 明示的な透過指定: アルファチャンネルが0より大きい部分を物体とする
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    elif channels == 4 and bg_mode == "auto":
        # 4チャンネルかつ auto の場合:
        # 書き出し時の数画素のゴミに惑わされないよう、全画素中の透明画素(<=10)の割合が0.1%以上ある場合のみ透過背景と判定
        alpha = img[:, :, 3]
        trans_pixels = np.count_nonzero(alpha <= 10)
        has_transparency = (trans_pixels / float(h * w)) >= 0.001
        if has_transparency:
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        else:
            # 不透明RGBAスプライトシート -> 色ベースの背景判定へフォールバック
            corners = np.array([
                img[0, 0, :3],
                img[0, -1, :3],
                img[-1, 0, :3],
                img[-1, -1, :3]
            ], dtype=np.int16)
            corner_range = np.max(corners, axis=0) - np.min(corners, axis=0)
            if int(np.max(corner_range)) <= tolerance:
                bg_bgr = np.median(corners, axis=0)
                diff = np.abs(img[:, :, :3].astype(np.int16) - bg_bgr)
                dist = np.max(diff, axis=2)
                mask = np.where(dist > tolerance, 255, 0).astype(np.uint8)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners_gray = [int(gray[0, 0]), int(gray[0, -1]), int(gray[-1, 0]), int(gray[-1, -1])]
                is_light_bg = (sum(corners_gray) / 4) > 127
                if is_light_bg:
                    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    elif bg_mode == "white":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    elif bg_mode == "black":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    elif bg_mode == "auto":
        if channels >= 3:
            corners = np.array([
                img[0, 0, :3],
                img[0, -1, :3],
                img[-1, 0, :3],
                img[-1, -1, :3]
            ], dtype=np.int16)
            corner_range = np.max(corners, axis=0) - np.min(corners, axis=0)
            if int(np.max(corner_range)) <= tolerance:
                # 四隅が同系色（マゼンタ、緑、白、黒などの単色背景）
                bg_bgr = np.median(corners, axis=0)
                diff = np.abs(img[:, :, :3].astype(np.int16) - bg_bgr)
                dist = np.max(diff, axis=2)
                mask = np.where(dist > tolerance, 255, 0).astype(np.uint8)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners_gray = [int(gray[0, 0]), int(gray[0, -1]), int(gray[-1, 0]), int(gray[-1, -1])]
                is_light_bg = (sum(corners_gray) / 4) > 127
                if is_light_bg:
                    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        else:
            corners_gray = [int(img[0, 0]), int(img[0, -1]), int(img[-1, 0]), int(img[-1, -1])]
            is_light_bg = (sum(corners_gray) / 4) > 127
            if is_light_bg:
                _, mask = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
            else:
                _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"未対応の bg_mode です: {bg_mode}")

    # 近接パーツを結合するための膨張処理（連結成分の判定にのみ使用し、
    # バウンディングボックス自体は元のマスクから測り直す）
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (padding * 2 + 1, padding * 2 + 1))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        dilated_mask = mask

    num_labels, labels = cv2.connectedComponents(dilated_mask)

    boxes = []
    if num_labels > 1:
        # 元マスクの前景画素 (mask > 0) のみを取り出してラベルごとに一括集計（高速化）
        ys, xs = np.nonzero(mask)
        if ys.size > 0:
            fg_labels = labels[ys, xs]
            valid = fg_labels > 0
            if np.any(valid):
                ys = ys[valid]
                xs = xs[valid]
                fg_labels = fg_labels[valid]

                min_x = np.full(num_labels, w, dtype=np.int32)
                max_x = np.full(num_labels, -1, dtype=np.int32)
                min_y = np.full(num_labels, h, dtype=np.int32)
                max_y = np.full(num_labels, -1, dtype=np.int32)

                np.minimum.at(min_x, fg_labels, xs)
                np.maximum.at(max_x, fg_labels, xs)
                np.minimum.at(min_y, fg_labels, ys)
                np.maximum.at(max_y, fg_labels, ys)

                for label in range(1, num_labels):
                    if max_x[label] < 0:
                        continue
                    x = int(min_x[label])
                    y = int(min_y[label])
                    bw = int(max_x[label] - x + 1)
                    bh = int(max_y[label] - y + 1)
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

    # prefix にディレクトリ区切りが含まれていても output_dir の外に書き出せないようにする
    safe_prefix = os.path.basename(prefix) or "sub_image"

    saved_files = []
    skipped = []
    used_filenames = set()
    for i, box in enumerate(boxes):
        raw_x = int(box.get("x", 0))
        raw_y = int(box.get("y", 0))
        bw = int(box.get("width", 0))
        bh = int(box.get("height", 0))

        # 範囲クリッピング（負座標分は右端・下端を保ったまま切り詰める）
        bx = max(0, raw_x)
        by = max(0, raw_y)
        bx2 = min(w, raw_x + bw)
        by2 = min(h, raw_y + bh)

        if bx2 <= bx or by2 <= by:
            skipped.append({"index": i, "reason": "out_of_bounds", "box": box})
            continue

        cropped = img[by:by2, bx:bx2]

        # box の index を優先使用し、数値化できない場合は i にフォールバック
        raw_idx = box.get("index", i)
        try:
            file_idx = int(raw_idx)
            base_filename = f"{safe_prefix}_{file_idx:03d}.png"
        except (ValueError, TypeError):
            base_filename = f"{safe_prefix}_{i:03d}.png"

        # ファイル名重複時の衝突回避 (_dup1, _dup2 ...)
        filename = base_filename
        dup_count = 1
        while filename in used_filenames:
            stem, ext = os.path.splitext(base_filename)
            filename = f"{stem}_dup{dup_count}{ext}"
            dup_count += 1
        used_filenames.add(filename)

        out_path = os.path.join(output_dir, filename)

        if save_image_safely(out_path, cropped):
            saved_files.append({
                "index": raw_idx,
                "path": out_path,
                "width": int(bx2 - bx),
                "height": int(by2 - by)
            })
        else:
            skipped.append({"index": i, "reason": "save_failed", "box": box})

    return {
        "saved_count": len(saved_files),
        "output_directory": os.path.abspath(output_dir),
        "files": saved_files,
        "skipped_count": len(skipped),
        "skipped": skipped
    }

@mcp.tool()
def preview_boxes(
    image_path: str,
    boxes: list[dict],
    output_path: str | None = None,
    line_thickness: int = 2,
    show_labels: bool = True,
    as_image: bool = False
) -> dict | Image:
    """
    画像上にバウンディングボックス（矩形枠とインデックス番号ラベル）を描画し、プレビュー画像を生成・保存します。
    検出結果やグリッド分割の確認・デバッグに利用できます。

    Parameters:
        image_path: 対象の画像ファイルパス
        boxes: バウンディングボックスのリスト (各要素は x, y, width, height, [index] を含む dict)
        output_path: プレビュー画像の保存先パス。省略時は元画像と同ディレクトリの '{basename}_preview.png'
        line_thickness: 描画する矩形枠の線の太さ(px)
        show_labels: インデックス番号ラベルを描画するかどうか
        as_image: True の場合、MCP クライアントでインライン表示可能な Image オブジェクトを返します
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    # グレースケールまたは 1 チャンネル画像の場合は枠線を鮮明に描画できるように BGR に変換
    if channels == 1 or (len(img.shape) == 3 and img.shape[2] == 1):
        preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        channels = 3
    else:
        preview = img.copy()

    box_color = (0, 255, 0, 255) if channels == 4 else (0, 255, 0)
    label_bg_color = (0, 0, 0, 200) if channels == 4 else (0, 0, 0)
    label_text_color = (255, 255, 255, 255) if channels == 4 else (255, 255, 255)

    drawn_count = 0
    for i, box in enumerate(boxes):
        bx = int(box.get("x", 0))
        by = int(box.get("y", 0))
        bw = int(box.get("width", 0))
        bh = int(box.get("height", 0))
        idx = box.get("index", i)

        if bw <= 0 or bh <= 0:
            continue

        cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), box_color, line_thickness)
        drawn_count += 1

        if show_labels:
            label = f"#{idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y1 = max(0, by - th - baseline - 4)
            label_y2 = label_y1 + th + baseline + 4
            label_x1 = max(0, bx)
            label_x2 = min(w, label_x1 + tw + 6)

            cv2.rectangle(preview, (label_x1, label_y1), (label_x2, label_y2), label_bg_color, -1)
            cv2.putText(
                preview,
                label,
                (label_x1 + 3, label_y1 + th + 2),
                font,
                font_scale,
                label_text_color,
                font_thickness,
                lineType=cv2.LINE_AA
            )

    if not output_path:
        dir_name = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}_preview.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not save_image_safely(output_path, preview):
        raise IOError(f"プレビュー画像の保存に失敗しました: {output_path}")

    if as_image:
        return Image(path=output_path)

    return {
        "preview_path": os.path.abspath(output_path),
        "drawn_boxes_count": drawn_count,
        "total_boxes_count": len(boxes),
        "image_size": {"width": int(w), "height": int(h)}
    }

@mcp.tool()
def view_region(
    image_path: str,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
    box: dict | None = None,
    scale: float = 1.0,
    output_path: str | None = None,
    as_image: bool = False
) -> dict | Image:
    """
    画像の指定された矩形領域を切り出し、必要に応じて拡大（scale）してプレビュー画像として保存します。
    特定のスプライトや細かいパーツの詳細確認に利用できます。

    Parameters:
        image_path: 対象の画像ファイルパス
        x: 切り出す領域の左端 x 座標
        y: 切り出す領域の上端 y 座標
        width: 切り出す領域の幅 (px)
        height: 切り出す領域の高さ (px)
        box: バウンディングボックス dict (x, y, width, height)。指定された場合は個別座標より優先
        scale: 拡大倍率 (例: 2.0 で 2倍、4.0 で 4倍。最近傍補間でドット絵もぼやけず拡大)
        output_path: 保存先パス。省略時は元画像と同ディレクトリの '{basename}_region_{x}_{y}.png'
        as_image: True の場合、MCP クライアントでインライン表示可能な Image オブジェクトを返します
    """
    img = load_image_safely(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    h, w = img.shape[:2]

    if box is not None:
        x = int(box.get("x", x))
        y = int(box.get("y", y))
        width = int(box.get("width", width))
        height = int(box.get("height", height))

    if width <= 0 or height <= 0:
        raise ValueError(f"width と height は正の整数で指定してください (width={width}, height={height})。")

    if scale <= 0:
        raise ValueError(f"scale は正の数値で指定してください (scale={scale})。")

    # 範囲クリッピング
    bx = max(0, x)
    by = max(0, y)
    bx2 = min(w, x + width)
    by2 = min(h, y + height)

    if bx2 <= bx or by2 <= by:
        raise ValueError(
            f"指定された領域が画像範囲外です: x={x}, y={y}, w={width}, h={height} (画像サイズ: {w}x{h})"
        )

    cropped = img[by:by2, bx:bx2]

    if scale != 1.0:
        new_w = max(1, int(round((bx2 - bx) * scale)))
        new_h = max(1, int(round((by2 - by) * scale)))
        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if not output_path:
        dir_name = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}_region_{bx}_{by}.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not save_image_safely(output_path, cropped):
        raise IOError(f"領域画像の保存に失敗しました: {output_path}")

    if as_image:
        return Image(path=output_path)

    return {
        "region_path": os.path.abspath(output_path),
        "original_region": {
            "x": int(bx),
            "y": int(by),
            "width": int(bx2 - bx),
            "height": int(by2 - by)
        },
        "output_size": {
            "width": int(cropped.shape[1]),
            "height": int(cropped.shape[0])
        },
        "scale": float(scale)
    }

def main():
    mcp.run()

if __name__ == "__main__":
    main()
