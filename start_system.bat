@echo off
REM トンネル地山評価システム起動スクリプト (Windows用)

echo 🚧 トンネル地山評価システムを起動します...

REM 仮想環境の確認
if not exist ".venv" (
    echo ❌ 仮想環境が見つかりません。
    echo 以下のコマンドで作成してください：
    echo   python -m venv .venv
    pause
    exit /b 1
)

REM 仮想環境を有効化
call .venv\Scripts\activate

REM Core APIの起動（新しいウィンドウで）
echo 📡 Core APIを起動中...
start "Core API" cmd /k "cd core && python tunnel_core_api.py"

REM 少し待機
timeout /t 3 /nobreak > nul

REM Streamlitダッシュボードの起動（新しいウィンドウで）
echo 🎯 ダッシュボードを起動中...
start "Streamlit Dashboard" cmd /k "streamlit run frontend\simple_dashboard.py"

echo.
echo ✅ システムが起動しました！
echo.
echo 📌 アクセスURL:
echo   - ダッシュボード: http://localhost:8501
echo   - API ドキュメント: http://localhost:8000/docs
echo.
echo 🛑 各ウィンドウで Ctrl+C を押して停止してください
echo.
pause