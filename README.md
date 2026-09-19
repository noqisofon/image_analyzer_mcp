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
5. **マルチプラットフォーム対応**:
   - Windows 環境での日本語を含むファイルパスにも対応。

---

## 提供ツール一覧

| ツール名 | 説明 | 主なパラメータ |
|---|---|---|
| `get_image_info` | 画像の全体サイズやメタデータを取得 | `image_path` |
| `find_sub_image_boxes` | スプライト・コラージュ要素の境界ボックスを検出 | `image_path`, `min_size` (最小サイズpx), `padding` (結合サイズpx), `bg_mode` (`auto`, `transparent`, `white`, `black`) |
| `get_grid_boxes` | 等間隔スプライトシートのグリッド分割 | `image_path`, `rows`/`cols` または `tile_width`/`tile_height`, `margin_x`, `margin_y`, `spacing_x`, `spacing_y` |
| `crop_and_save_sub_images` | ボックス範囲で切り出して連番保存 | `image_path`, `boxes`, `output_dir`, `prefix` |

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
