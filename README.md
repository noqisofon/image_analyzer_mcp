# image-analyzer-mcp

AI エージェント（Antigravity、Claude 等）が画像の全体サイズや、スプライトシート・コラージュ内の各要素の境界・サイズ（バウンディングボックス）をピクセル精度で素早く把握・切り出しできるように支援する MCP (Model Context Protocol) サーバーです。

## 主な機能

1. **`get_image_info`**:
   - 画像の全体幅・高さ・チャンネル数（アルファ有無）・ファイルサイズを軽量かつ確実に取得。
2. **`find_sub_image_boxes`**:
   - 透過 PNG や白/黒/自動背景判定による輪郭抽出。
   - 近接パーツの結合パディングや、上から下・左から右への自然なソート。
3. **`get_grid_boxes`**:
   - ゲームスプライトシートなどの規則的なタイル配置をグリッド分割（行・列数指定 または タイル幅・高さ指定）。
4. **`crop_and_save_sub_images`**:
   - 検出・指定したバウンディングボックスを元に、個別の画像を切り出して連番で指定フォルダに保存。
5. **`preview_boxes`**:
   - 検出したバウンディングボックスを画像上に枠線およびインデックス番号付きで描画し、可視化プレビュー画像を生成。
6. **`view_region`**:
   - 指定した領域（座標または box）を切り出し、必要に応じて拡大（scale）して確認用画像を保存。
7. **マルチプラットフォーム対応**:
   - Windows 環境での日本語を含むファイルパスにも対応。

---

## 提供ツール一覧

| ツール名                   | 説明                                           | 主なパラメータ                                                                                                         |
| -------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `get_image_info`           | 画像の全体サイズやメタデータを取得             | `image_path`                                                                                                           |
| `find_sub_image_boxes`     | スプライト・コラージュ要素の境界ボックスを検出 | `image_path`, `min_size` (最小サイズpx、既定8), `padding` (結合サイズpx), `bg_mode` (`auto`, `transparent`, `white`, `black`, `color`), `bg_color` ([R, G, B]), `tolerance` (許容色差), `merged_threshold_ratio` (融合警告の中央値比閾値、既定20.0) |
| `get_grid_boxes`           | 等間隔スプライトシートのグリッド分割           | `image_path`, `rows`/`cols` または `tile_width`/`tile_height`, `margin_x`, `margin_y`, `spacing_x`, `spacing_y`        |
| `crop_and_save_sub_images` | ボックス範囲で切り出して連番保存               | `image_path`, `boxes`, `output_dir`, `prefix`, `overwrite` (上書き可否、デフォルトFalse)                              |
| `preview_boxes`            | 検出ボックスを画像上に枠線・ラベル付き描画     | `image_path`, `boxes`, `output_path`, `line_thickness`, `show_labels`, `as_image` (Image返却)          |
| `view_region`              | 指定領域を切り出し・拡大プレビュー保存         | `image_path`, `x`/`y`/`width`/`height` または `box`, `scale`, `output_path`, `as_image` (Image返却)    |

### `find_sub_image_boxes` の利用ガイダンスと注意点

- **`min_size`（最小サイズ）の既定値**:
  - デフォルトは `8` です。レトロゲームや 16px タイルセット文化における小さな小物スプライト（8x8 や 8x14 など）を自然に検出できるように設定されています。
  - 透過画像の影・アンチエイリアスの微小な残骸やノイズを除外したい場合は、`min_size=16` や `32` などの大きめの値を明示的に指定してください。
- **融合警告フラグ (`warning`)**:
  - `entire_image_or_wrong_bg`: ボックス面積が画像全体の 90% 以上を占めている場合に付与されます（タイルメッシュ全体が連結しているか、背景色指定が合っていない可能性があります）。中央値計算からは自動的に除外されます。
  - `possible_merged_components`: 有効ボックスが 5 個以上ある場合、中央値の `merged_threshold_ratio`（既定 20 倍）を超える異常に巨大なボックスに付与されます（密集した家具や小物が 1 つに融合している可能性があります）。
  - **重要: 警告なし ＝ 完全に単一オブジェクトに分離されている保証ではありません**。数個の小物が融合している場合（面積比が 4〜5 倍程度など）は面積比の閾値だけでは検知できません。密集したスプライトシートでは `preview_boxes` による目視確認や、規則正しいタイルの場合は `get_grid_boxes` の併用を推奨します。
- **`preview_boxes` の可視化特徴**:
  - 通常ボックスは緑枠線（`#idx`）、警告付きボックスは**太枠シアン線（`!#idx`）**で描画され、一目で問題箇所を把握できます。
  - 不透明な帯ではなく、黒縁取り＋白文字による 2 パス描画を採用しているため、文字が重なっても背後のドット絵を覆い隠しません。


---

## MCP クライアントへの登録方法

### 1. Antigravity / Claude Desktop / Cursor の設定例

MCP 設定ファイル（例: `~/.gemini/antigravity/mcp_config.json` や Claude Desktop の `claude_desktop_config.json`）の `mcpServers` に以下を追加します：

```json
{
  "mcpServers": {
    "image-analyzer": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/Users/nedri/workspace/image_analyzer_mcp",
        "run",
        "image-analyzer-mcp"
      ]
    }
  }
}
```

または Python スクリプト直接指定の場合：

```json
{
  "mcpServers": {
    "image-analyzer": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/Users/nedri/workspace/image_analyzer_mcp",
        "run",
        "python",
        "-m",
        "image_analyzer_mcp"
      ]
    }
  }
}
```

---

## 開発・テスト

```bash
# 依存関係のインストール
uv sync

# エントリポイントの動作確認
uv run python -c "from image_analyzer_mcp import mcp; print(mcp.name)"
```
