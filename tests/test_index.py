import os

import cv2
import numpy as np
import pytest

from image_analyzer_mcp.index import (
    crop_and_save_sub_images,
    find_sub_image_boxes,
    preview_boxes,
    view_region,
)


def _save(tmp_path, img, name="test.png"):
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def test_padding_does_not_inflate_bbox_of_isolated_sprite(tmp_path):
    # 透明キャンバス上に孤立した 10x10 の不透明スプライトを1つ配置する。
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[10:20, 10:20] = (255, 255, 255, 255)
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=1, padding=2)

    assert result["count"] == 1
    box = result["boxes"][0]
    assert (box["x"], box["y"], box["width"], box["height"]) == (10, 10, 10, 10)


def test_padding_still_merges_nearby_parts(tmp_path):
    # 2px 離れた2つのパーツは padding=2 で1つのボックスに結合される。
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[10:20, 10:20] = (255, 255, 255, 255)
    img[10:20, 22:32] = (255, 255, 255, 255)
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=1, padding=2)

    assert result["count"] == 1
    box = result["boxes"][0]
    assert (box["x"], box["y"], box["width"], box["height"]) == (10, 10, 22, 10)


def test_padding_zero_keeps_parts_separate(tmp_path):
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[10:20, 10:20] = (255, 255, 255, 255)
    img[10:20, 22:32] = (255, 255, 255, 255)
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=1, padding=0)

    assert result["count"] == 2


def test_row_sort_is_stable_with_uneven_row_boundaries(tmp_path):
    # 同じ行内でも y に若干のブレがある3スプライト + 次の行に1スプライト。
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    img[10:30, 10:30] = (255, 255, 255, 255)   # row0, x=10
    img[15:35, 60:80] = (255, 255, 255, 255)   # row0 (yブレあり), x=60
    img[12:32, 110:130] = (255, 255, 255, 255)  # row0 (yブレあり), x=110
    img[80:100, 10:30] = (255, 255, 255, 255)  # row1, x=10
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=1, padding=1)

    assert result["count"] == 4
    xs_in_order = [b["x"] for b in result["boxes"]]
    # 1行目の3つは x の昇順、続いて2行目が最後に来る。
    assert xs_in_order == [10, 60, 110, 10]
    assert [b["index"] for b in result["boxes"]] == [0, 1, 2, 3]


def test_transparent_mode_without_alpha_raises(tmp_path):
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    path = _save(tmp_path, img)

    with pytest.raises(ValueError, match="transparent"):
        find_sub_image_boxes(path, bg_mode="transparent")


def test_transparent_mode_with_alpha_still_works(tmp_path):
    img = np.zeros((50, 50, 4), dtype=np.uint8)
    img[5:15, 5:15] = (255, 255, 255, 255)
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=1, bg_mode="transparent")

    assert result["count"] == 1


def test_crop_negative_coordinates_are_clipped_not_shifted(tmp_path):
    # x=-5, width=10 の場合、画像内に見えているのは x=0..5 の範囲のみのはず。
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    path = _save(tmp_path, img)
    out_dir = tmp_path / "out"

    result = crop_and_save_sub_images(
        path,
        boxes=[{"x": -5, "y": 0, "width": 10, "height": 10}],
        output_dir=str(out_dir),
    )

    assert result["saved_count"] == 1
    saved = result["files"][0]
    assert saved["width"] == 5
    assert saved["height"] == 10


def test_crop_prefix_cannot_escape_output_dir(tmp_path):
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    path = _save(tmp_path, img)
    out_dir = tmp_path / "out"
    escape_target = tmp_path / "evil_000.png"

    result = crop_and_save_sub_images(
        path,
        boxes=[{"x": 0, "y": 0, "width": 10, "height": 10}],
        output_dir=str(out_dir),
        prefix="../evil",
    )

    assert result["saved_count"] == 1
    saved_path = result["files"][0]["path"]
    assert os.path.commonpath([saved_path, str(out_dir)]) == str(out_dir)
    assert not escape_target.exists()


def test_crop_out_of_range_box_is_reported_as_skipped(tmp_path):
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    path = _save(tmp_path, img)
    out_dir = tmp_path / "out"

    result = crop_and_save_sub_images(
        path,
        boxes=[
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 1000, "y": 1000, "width": 10, "height": 10},
        ],
        output_dir=str(out_dir),
    )

    assert result["saved_count"] == 1
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["index"] == 1
    assert result["skipped"][0]["reason"] == "out_of_bounds"


def test_magenta_background_auto_detection(tmp_path):
    # マゼンタ背景 (BGR: 255, 0, 255) に緑のスプライト (BGR: 0, 255, 0)
    img = np.full((100, 100, 3), (255, 0, 255), dtype=np.uint8)
    img[20:40, 20:50] = (0, 255, 0)
    path = _save(tmp_path, img)

    result = find_sub_image_boxes(path, min_size=5, bg_mode="auto")

    assert result["count"] == 1
    box = result["boxes"][0]
    assert (box["x"], box["y"], box["width"], box["height"]) == (20, 20, 30, 20)


def test_explicit_bg_color_and_validation(tmp_path):
    # シアン背景 (RGB: 0, 255, 255 / BGR: 255, 255, 0)
    img = np.full((100, 100, 3), (255, 255, 0), dtype=np.uint8)
    img[10:30, 10:30] = (0, 0, 255)
    path = _save(tmp_path, img)

    # RGB [0, 255, 255] を指定
    result = find_sub_image_boxes(path, min_size=5, bg_color=[0, 255, 255])
    assert result["count"] == 1
    box = result["boxes"][0]
    assert (box["x"], box["y"], box["width"], box["height"]) == (10, 10, 20, 20)

    # bg_mode="color" で bg_color 未指定時はエラー
    with pytest.raises(ValueError, match="bg_color"):
        find_sub_image_boxes(path, bg_mode="color")

    # bg_color が3要素でない場合もエラー
    with pytest.raises(ValueError, match="3要素"):
        find_sub_image_boxes(path, bg_color=[255, 255])


def test_main_module_importable():
    import image_analyzer_mcp.__main__ as main_mod

    assert hasattr(main_mod, "main")


def test_preview_boxes_draws_and_saves(tmp_path):
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    path = _save(tmp_path, img, "source.png")

    boxes = [
        {"x": 10, "y": 10, "width": 30, "height": 30, "index": 0},
        {"x": 50, "y": 50, "width": 20, "height": 20, "index": 1},
    ]

    result = preview_boxes(path, boxes)

    assert result["drawn_boxes_count"] == 2
    assert result["total_boxes_count"] == 2
    assert os.path.isfile(result["preview_path"])

    # プレビュー画像が生成され、サイズが元画像と同一か検証
    preview_img = cv2.imread(result["preview_path"])
    assert preview_img.shape[:2] == (100, 100)


def test_preview_boxes_with_alpha_and_custom_output(tmp_path):
    img = np.zeros((80, 80, 4), dtype=np.uint8)
    path = _save(tmp_path, img, "alpha.png")
    custom_out = str(tmp_path / "custom_preview.png")

    boxes = [{"x": 10, "y": 10, "width": 20, "height": 20}]
    result = preview_boxes(path, boxes, output_path=custom_out, show_labels=False)

    assert result["preview_path"] == os.path.abspath(custom_out)
    assert os.path.isfile(custom_out)


def test_view_region_crop_and_scale(tmp_path):
    # 100x100 画像で (20, 30) から 10x10 の赤い四角形
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:40, 20:30] = (0, 0, 255)
    path = _save(tmp_path, img, "region_src.png")

    # scale=1.0 で切り出し
    res1 = view_region(path, x=20, y=30, width=10, height=10, scale=1.0)
    assert res1["output_size"] == {"width": 10, "height": 10}
    assert os.path.isfile(res1["region_path"])
    cropped1 = cv2.imread(res1["region_path"])
    assert cropped1.shape == (10, 10, 3)
    assert np.all(cropped1 == (0, 0, 255))

    # scale=3.0 で拡大切り出し
    res2 = view_region(path, x=20, y=30, width=10, height=10, scale=3.0)
    assert res2["output_size"] == {"width": 30, "height": 30}
    cropped2 = cv2.imread(res2["region_path"])
    assert cropped2.shape == (30, 30, 3)
    assert np.all(cropped2 == (0, 0, 255))


def test_view_region_box_dict_and_validation(tmp_path):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    path = _save(tmp_path, img, "val_src.png")

    # box dict での指定
    res = view_region(path, box={"x": 5, "y": 5, "width": 10, "height": 10})
    assert res["output_size"] == {"width": 10, "height": 10}

    # 範囲外の指定で ValueError
    with pytest.raises(ValueError, match="画像範囲外"):
        view_region(path, x=100, y=100, width=10, height=10)

    # 不正な幅・高さで ValueError
    with pytest.raises(ValueError, match="正の整数"):
        view_region(path, x=0, y=0, width=0, height=10)

    # 不正な scale で ValueError
    with pytest.raises(ValueError, match="正の数値"):
        view_region(path, x=0, y=0, width=10, height=10, scale=-1.0)


def test_16bit_png_normalization_and_detection(tmp_path):
    # 16-bit Grayscale PNG (uint16)
    img16 = np.zeros((60, 60), dtype=np.uint16)
    img16[15:35, 15:35] = 65000  # 白に近い
    path16 = _save(tmp_path, img16, "test16_gray.png")

    res = find_sub_image_boxes(path16, min_size=5, bg_mode="auto")
    assert res["count"] == 1
    box = res["boxes"][0]
    assert (box["x"], box["y"], box["width"], box["height"]) == (15, 15, 20, 20)

    # 16-bit RGBA PNG
    img16_rgba = np.zeros((60, 60, 4), dtype=np.uint16)
    img16_rgba[15:35, 15:35] = (65535, 65535, 65535, 65535)
    path16_rgba = _save(tmp_path, img16_rgba, "test16_rgba.png")

    res_rgba = find_sub_image_boxes(path16_rgba, min_size=5, bg_mode="auto")
    assert res_rgba["count"] == 1
    assert res_rgba["boxes"][0]["width"] == 20


def test_opaque_rgba_with_solid_background_fallback(tmp_path):
    # 4チャンネル (RGBA) だがアルファが全面不透明 (255) で背景が白 (255, 255, 255)
    # スプライトとして黒い四角を2つ配置
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    img[10:30, 10:30, :3] = 0  # sprite 1 (black)
    img[50:70, 50:70, :3] = 0  # sprite 2 (black)
    path = _save(tmp_path, img, "opaque_rgba.png")

    # bg_mode="auto" で全面1つの巨大boxにならず、2つのスプライトが検出されること
    res = find_sub_image_boxes(path, min_size=5, bg_mode="auto")
    assert res["count"] == 2
    assert res["boxes"][0]["width"] == 20
    assert res["boxes"][1]["width"] == 20


def test_opaque_rgba_with_garbage_transparent_pixels(tmp_path):
    # 100x100 (10000ピクセル) 中、書き出しゴミとして5ピクセルだけアルファ=0が存在 (0.05% < 0.1%)
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    img[10:30, 10:30, :3] = 0  # sprite
    img[0, 0:5, 3] = 0         # 5ピクセルの透明ゴミ
    path = _save(tmp_path, img, "garbage_rgba.png")

    # 0.1% 未満のゴミに惑わされず色ベース判定にフォールバックすること
    res = find_sub_image_boxes(path, min_size=5, bg_mode="auto")
    assert res["count"] == 1
    assert res["boxes"][0]["width"] == 20


def test_crop_index_preservation_and_collision_resolution(tmp_path):
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    path = _save(tmp_path, img, "crop_src.png")
    out_dir = tmp_path / "crop_out"

    # index 3 と 7、衝突用 index 3、負数(-1)、小数(3.9)、文字列("bad")、そして正規の index 5 を追加
    boxes = [
        {"x": 10, "y": 10, "width": 10, "height": 10, "index": 3},
        {"x": 30, "y": 30, "width": 10, "height": 10, "index": 7},
        {"x": 50, "y": 50, "width": 10, "height": 10, "index": 3},     # 重複 -> _dup1
        {"x": 10, "y": 70, "width": 10, "height": 10, "index": -1},    # 負数 -> _i003
        {"x": 30, "y": 70, "width": 10, "height": 10, "index": 3.9},   # 小数 -> _i004
        {"x": 50, "y": 70, "width": 10, "height": 10, "index": "bad"},  # 不正文字列 -> _i005
        {"x": 70, "y": 70, "width": 10, "height": 10, "index": 5},     # 正規の index 5 -> sprite_005 (横取りされず取得可能！)
    ]

    res = crop_and_save_sub_images(path, boxes, output_dir=str(out_dir), prefix="sprite")

    assert res["saved_count"] == 7
    filenames = [os.path.basename(f["path"]) for f in res["files"]]
    assert filenames == [
        "sprite_003.png",
        "sprite_007.png",
        "sprite_003_dup1.png",
        "sprite_i003.png",
        "sprite_i004.png",
        "sprite_i005.png",
        "sprite_005.png",
    ]
    assert len(set(filenames)) == 7

    # 2回目の実行 (overwrite=False: デフォルト): ディスク上の既存ファイルと衝突せず _dup が付くこと
    res_second = crop_and_save_sub_images(
        path,
        boxes=[{"x": 10, "y": 10, "width": 10, "height": 10, "index": 3}],
        output_dir=str(out_dir),
        prefix="sprite"
    )
    second_filename = os.path.basename(res_second["files"][0]["path"])
    assert second_filename == "sprite_003_dup2.png"

    # 3回目の実行 (overwrite=True): 既存ファイルに直接上書きされること
    res_overwrite = crop_and_save_sub_images(
        path,
        boxes=[{"x": 10, "y": 10, "width": 10, "height": 10, "index": 3}],
        output_dir=str(out_dir),
        prefix="sprite",
        overwrite=True
    )
    overwrite_filename = os.path.basename(res_overwrite["files"][0]["path"])
    assert overwrite_filename == "sprite_003.png"


def test_preview_boxes_grayscale_color_boxes(tmp_path):
    # 1チャンネル (H, W) グレースケール画像
    gray = np.zeros((60, 60), dtype=np.uint8)
    path = _save(tmp_path, gray, "gray.png")

    boxes = [{"x": 10, "y": 10, "width": 20, "height": 20, "index": 0}]
    res = preview_boxes(path, boxes, show_labels=False)

    preview_img = cv2.imread(res["preview_path"])
    # 出力は 3 チャンネル (BGR) に変換されていること
    assert len(preview_img.shape) == 3 and preview_img.shape[2] == 3

    # 枠線 (10, 10) のピクセルが緑 (BGR: 0, 255, 0) であること (黒 0 ではないこと)
    pixel = preview_img[10, 10]
    assert pixel[1] > 200  # G が高い
    assert pixel[0] == 0 and pixel[2] == 0  # B, R は 0


def test_as_image_return(tmp_path):
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    path = _save(tmp_path, img, "as_image_src.png")

    res_preview = preview_boxes(
        path,
        boxes=[{"x": 5, "y": 5, "width": 10, "height": 10}],
        as_image=True
    )
    # Image オブジェクトが返ること
    assert hasattr(res_preview, "path")
    assert os.path.isfile(res_preview.path)

    res_region = view_region(path, x=5, y=5, width=10, height=10, as_image=True)
    assert hasattr(res_region, "path")
    assert os.path.isfile(res_region.path)


def _reference_find_boxes(mask, padding, min_size):
    """旧実装（全画素走査方式）による基準バウンディングボックス計算"""
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
            boxes.append({"x": x, "y": y, "width": bw, "height": bh})
    return sorted(boxes, key=lambda b: (b["y"], b["x"]))


def test_roi_consistency_with_reference(tmp_path):
    # 1. 透過PNG: 画像端に接するスプライト（左上・右上・左下・右下）を含めた配置
    np.random.seed(42)
    img_alpha = np.zeros((300, 300, 4), dtype=np.uint8)
    img_alpha[0:20, 0:20] = (255, 255, 255, 255)       # 左上端
    img_alpha[0:20, 280:300] = (255, 255, 255, 255)   # 右上端
    img_alpha[280:300, 0:20] = (255, 255, 255, 255)   # 左下端
    img_alpha[280:300, 280:300] = (255, 255, 255, 255) # 右下端
    for _ in range(25):
        w = np.random.randint(8, 25)
        h = np.random.randint(8, 25)
        x = np.random.randint(0, 300 - w)
        y = np.random.randint(0, 300 - h)
        img_alpha[y:y+h, x:x+w] = (255, 255, 255, 255)

    path_alpha = _save(tmp_path, img_alpha, "roi_alpha.png")

    for pad in [0, 1, 3]:
        roi_result = find_sub_image_boxes(path_alpha, min_size=8, padding=pad)
        alpha = img_alpha[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        ref_boxes = _reference_find_boxes(mask, padding=pad, min_size=8)

        # 個数と全ボックスの座標・サイズの一致
        assert roi_result["count"] == len(ref_boxes)
        roi_sorted = sorted(roi_result["boxes"], key=lambda b: (b["y"], b["x"]))
        for b_roi, b_ref in zip(roi_sorted, ref_boxes):
            assert (b_roi["x"], b_roi["y"], b_roi["width"], b_roi["height"]) == \
                   (b_ref["x"], b_ref["y"], b_ref["width"], b_ref["height"])

    # 2. マゼンタ単色背景 (BGR: 255, 0, 255) で端点接地スプライトの検証 (四隅頂点以外の端に配置)
    img_mag = np.full((200, 200, 3), (255, 0, 255), dtype=np.uint8)
    img_mag[0:15, 5:25] = (0, 255, 0)          # 上端に接地
    img_mag[50:70, 0:20] = (0, 255, 0)          # 左端に接地
    img_mag[100:130, 120:150] = (0, 0, 255)
    path_mag = _save(tmp_path, img_mag, "roi_magenta.png")
    res_mag = find_sub_image_boxes(path_mag, min_size=5, bg_mode="auto")
    assert res_mag["count"] == 3
    # 上端接地の y=0 と左端接地の x=0 が正確に取れること
    ys = [b["y"] for b in res_mag["boxes"]]
    xs = [b["x"] for b in res_mag["boxes"]]
    assert 0 in ys
    assert 0 in xs


def test_find_sub_image_boxes_default_min_size(tmp_path):
    # min_size 未指定（デフォルト8）で 10x10 は検出され、6x6 は除外される
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[10:20, 10:20] = (255, 255, 255, 255)  # 10x10
    img[40:46, 40:46] = (255, 255, 255, 255)  # 6x6
    path = _save(tmp_path, img, "min_size.png")

    res = find_sub_image_boxes(path, padding=0)
    assert res["count"] == 1
    assert res["boxes"][0]["width"] == 10
    assert res["boxes"][0]["height"] == 10


def test_merged_components_warning_on_large_canvas(tmp_path):
    # 1024x1024 の広いキャンバス: 16x16 の小物が6個 (中央値 256) と 80x80 の融合ボックス (面積 6400 = 25倍)
    # 画面占有率は 6400 / 1048576 = 0.61%（5%ルールでは見逃すケース）だが中央値比で確実に警告が付く
    img = np.zeros((1024, 1024, 4), dtype=np.uint8)
    for i in range(6):
        x = 50 + i * 40
        img[50:66, x:x+16] = (255, 255, 255, 255)
    img[300:380, 300:380] = (255, 255, 255, 255)  # 80x80
    path = _save(tmp_path, img, "large_canvas.png")

    res = find_sub_image_boxes(path, padding=0)
    assert res["count"] == 7
    assert res["median_area"] == 256.0
    assert res["warnings_count"] == 1

    large_box = [b for b in res["boxes"] if b["width"] == 80 and b["height"] == 80][0]
    assert large_box["area_ratio_to_median"] == 25.0
    assert large_box["warning"] == "possible_merged_components"

    small_boxes = [b for b in res["boxes"] if b["width"] == 16 and b["height"] == 16]
    for sb in small_boxes:
        assert "warning" not in sb
        assert sb["area_ratio_to_median"] == 1.0


def test_no_false_positive_warning_on_single_large_sprite(tmp_path):
    # 256x256 キャンバス: 64x64 の単体大物スプライト (面積 4096 = 画面占有率 6.25%) と 16x16 の小物 6個
    # 画面占有率 5% ルールでは誤判定されるが、中央値比は 4096 / 256 = 16.0 (< 20.0) のため誤警告が付かない
    img = np.zeros((256, 256, 4), dtype=np.uint8)
    for i in range(6):
        x = 20 + i * 30
        img[20:36, x:x+16] = (255, 255, 255, 255)
    img[100:164, 100:164] = (255, 255, 255, 255)  # 64x64
    path = _save(tmp_path, img, "single_large.png")

    res = find_sub_image_boxes(path, padding=0)
    assert res["count"] == 7
    assert res["median_area"] == 256.0
    assert res["warnings_count"] == 0

    large_box = [b for b in res["boxes"] if b["width"] == 64 and b["height"] == 64][0]
    assert large_box["area_ratio_to_median"] == 16.0
    assert "warning" not in large_box


def test_entire_image_or_wrong_bg_warning(tmp_path):
    # 300x300 キャンバス (90000 px²): 285x285 の巨大ボックス (81225 px² = 90.25%) と 9x9 の小物 5個
    # 巨大ボックスは画面90%以上の警告を受け、中央値計算から除外される
    img = np.zeros((300, 300, 4), dtype=np.uint8)
    img[0:285, 0:285] = (255, 255, 255, 255)  # 285x285
    for i in range(5):
        y = 10 + i * 30
        img[y:y+9, 290:299] = (255, 255, 255, 255)  # 9x9
    path = _save(tmp_path, img, "giant_box.png")

    res = find_sub_image_boxes(path, padding=0)
    assert res["count"] == 6
    assert res["median_area"] == 81.0
    assert res["warnings_count"] == 1

    giant_box = [b for b in res["boxes"] if b["width"] == 285 and b["height"] == 285][0]
    assert giant_box["warning"] == "entire_image_or_wrong_bg"


def test_custom_merged_threshold_ratio(tmp_path):
    # 10x10 の小物 5個 (中央値 100) と 22x22 の小規模融合ボックス (面積 484 = 4.84倍)
    # デフォルト (20.0) では警告されないが、merged_threshold_ratio=4.0 を指定すると警告が付与される
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    for i in range(5):
        x = 10 + i * 20
        img[10:20, x:x+10] = (255, 255, 255, 255)
    img[60:82, 60:82] = (255, 255, 255, 255)  # 22x22
    path = _save(tmp_path, img, "threshold_param.png")

    # デフォルト
    res_def = find_sub_image_boxes(path, padding=0)
    target_box = [b for b in res_def["boxes"] if b["width"] == 22 and b["height"] == 22][0]
    assert target_box["area_ratio_to_median"] == 4.84
    assert "warning" not in target_box

    # merged_threshold_ratio=4.0
    res_custom = find_sub_image_boxes(path, padding=0, merged_threshold_ratio=4.0)
    target_box_warned = [b for b in res_custom["boxes"] if b["width"] == 22 and b["height"] == 22][0]
    assert target_box_warned["warning"] == "possible_merged_components"
    assert res_custom["warnings_count"] == 1


def test_preview_boxes_cyan_warning_and_styling(tmp_path):
    # 通常ボックス (緑) と警告付きボックス (シアン) を含むプレビュー描画テスト
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = _save(tmp_path, img, "preview_test.png")
    out_path = str(tmp_path / "out_preview.png")

    boxes = [
        {"x": 10, "y": 10, "width": 20, "height": 20, "index": 0},
        {"x": 50, "y": 50, "width": 30, "height": 30, "index": 1, "warning": "possible_merged_components"}
    ]
    res = preview_boxes(path, boxes, output_path=out_path, show_labels=True)
    assert os.path.isfile(out_path)

    preview_img = cv2.imread(out_path)
    # 通常ボックス (x=10..30, y=10) の枠線上の画素に緑 (0, 255, 0) が含まれること
    assert np.any(preview_img[10, 10:30] == (0, 255, 0))
    # 警告ボックス (x=50..80, y=50) の枠線上の画素にシアン (255, 255, 0) が含まれること
    assert np.any(preview_img[50, 50:80] == (255, 255, 0))






