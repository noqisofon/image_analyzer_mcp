import cv2
import numpy as np
import pytest

from image_analyzer_mcp.index import find_sub_image_boxes


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
