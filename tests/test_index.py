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


