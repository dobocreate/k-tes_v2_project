#!/bin/bash

# トンネル地山評価システム起動スクリプト

echo "🚧 トンネル地山評価システムを起動します..."

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 仮想環境の確認
if [ ! -d ".venv" ]; then
    echo "❌ 仮想環境が見つかりません。"
    echo "以下のコマンドで作成してください："
    echo "  python3 -m venv .venv"
    exit 1
fi

# 仮想環境を有効化
source .venv/bin/activate

# Core APIの起動（バックグラウンド）
echo "📡 Core APIを起動中..."
cd core
python tunnel_core_api.py &
API_PID=$!
cd ..

# 少し待機
sleep 3

# Streamlitダッシュボードの起動
echo "🎯 ダッシュボードを起動中..."
streamlit run frontend/simple_dashboard.py &
STREAMLIT_PID=$!

echo ""
echo "✅ システムが起動しました！"
echo ""
echo "📌 アクセスURL:"
echo "  - ダッシュボード: http://localhost:8501"
echo "  - API ドキュメント: http://localhost:8000/docs"
echo ""
echo "🛑 停止するには Ctrl+C を押してください"

# 終了処理
trap "echo ''; echo '👋 システムを停止します...'; kill $API_PID $STREAMLIT_PID; exit" INT

# プロセスが終了するまで待機
wait