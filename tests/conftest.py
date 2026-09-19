"""
現在の環境では mcp パッケージが Python 3.14rc2 + pydantic の組み合わせで
インポート時に壊れる（本リポジトリの画像処理ロジックとは無関係の依存関係の問題）。
image_analyzer_mcp.index は FastMCP の @mcp.tool() デコレータ経由でしか
mcp を使わないため、テストではそこだけをスタブ化してロジックを検証する。
"""
import sys
import types


def _install_fake_mcp():
    if "mcp.server.mcpserver" in sys.modules:
        return

    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    mcpserver_mod = types.ModuleType("mcp.server.mcpserver")

    class _FakeMCPServer:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def run(self, *args, **kwargs):
            pass

    class _FakeImage:
        def __init__(self, path=None, data=None, format=None):
            self.path = path
            self.data = data
            self.format = format

    mcpserver_mod.MCPServer = _FakeMCPServer
    mcpserver_mod.Image = _FakeImage
    mcp_mod.server = server_mod
    server_mod.mcpserver = mcpserver_mod

    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.mcpserver"] = mcpserver_mod


_install_fake_mcp()
