"""
Streamlitなしでの動作確認用スクリプト
"""
import http.server
import socketserver
import os

PORT = 8501

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>トンネル地山評価システム - テスト</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f0f2f6;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1f77b4;
        }}
        .status {{
            padding: 20px;
            background: #e8f4f8;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .info {{
            color: #666;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚧 トンネル地山評価システム</h1>
        <div class="status">
            <p class="success">✅ Webサーバーは正常に動作しています！</p>
            <p>ポート {PORT} でリクエストを受け付けています。</p>
        </div>
        <div class="info">
            <h2>次のステップ</h2>
            <ol>
                <li>Streamlitをインストールしてください：
                    <pre>pip install streamlit</pre>
                </li>
                <li>メインアプリケーションを起動してください：
                    <pre>streamlit run simple_dashboard.py</pre>
                </li>
            </ol>
            <p>このページが表示されていれば、Pythonとネットワーク設定は正常です。</p>
        </div>
    </div>
</body>
</html>
"""

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

print(f"🚧 テストサーバーを起動しています...")
print(f"📍 http://localhost:{PORT} でアクセスしてください")
print("終了するには Ctrl+C を押してください")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 サーバーを停止しました")