# トンネル地山評価システム - 起動ガイド

このガイドでは、トンネル地山評価システムの起動方法を説明します。

## 📍 プロジェクトディレクトリ

本システムのプロジェクトディレクトリは以下の場所にあります：
```
/mnt/c/users/kishida/cursorproject/k-tes_v2_project
```

### ディレクトリ構造
```
/mnt/c/users/kishida/cursorproject/k-tes_v2_project/
├── core/                    # Core API
│   ├── tunnel_core_api.py   # APIメインファイル
│   ├── requirements.txt     # API依存関係
│   └── models/             # モデルファイル配置場所
├── frontend/               # フロントエンド
│   ├── simple_dashboard.py  # ダッシュボードメインファイル
│   └── requirements.txt    # UI依存関係
├── docs/                   # ドキュメント
│   └── startup-guide.md    # このファイル
├── .venv/                  # Python仮想環境
├── start_system.sh         # Linux/Mac用起動スクリプト
└── start_system.bat        # Windows用起動スクリプト
```

## 📋 前提条件

- Python 3.9以上がインストールされていること
- プロジェクトの依存関係がインストールされていること
- 上記プロジェクトディレクトリにアクセスできること

## 🚀 クイックスタート

システムを起動するには、2つのコンポーネントを起動する必要があります：
1. Core API（バックエンド）
2. Streamlitダッシュボード（フロントエンド）

### ステップ1: Core APIの起動

新しいターミナルウィンドウを開き、以下のコマンドを実行します：

```bash
# プロジェクトディレクトリに移動
cd /mnt/c/users/kishida/cursorproject/k-tes_v2_project

# 仮想環境を有効化
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core APIを起動
cd core
python tunnel_core_api.py
```

**起動成功時の表示：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started server process [...]
INFO:     Application startup complete.
```

### ステップ2: Streamlitダッシュボードの起動

別の新しいターミナルウィンドウを開き、以下のコマンドを実行します：

```bash
# プロジェクトディレクトリに移動
cd /mnt/c/users/kishida/cursorproject/k-tes_v2_project

# 仮想環境を有効化
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# ダッシュボードを起動
streamlit run frontend/simple_dashboard.py
```

**起動成功時の表示：**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://xxx.xxx.xxx.xxx:8501
```

## 🌐 アクセス方法

両方のサービスが起動したら、以下のURLにアクセスできます：

- **ダッシュボード**: http://localhost:8501
- **API ドキュメント**: http://localhost:8000/docs

## 🔧 詳細な起動オプション

### Core APIの起動オプション

#### 標準起動
```bash
python tunnel_core_api.py
```

#### カスタムポートで起動
```bash
# ポート8001で起動
uvicorn tunnel_core_api:app --port 8001 --reload
```

#### 環境変数を使用した起動
```bash
# .envファイルを作成している場合
API_PORT=8001 python tunnel_core_api.py
```

### Streamlitダッシュボードの起動オプション

#### 標準起動
```bash
streamlit run frontend/simple_dashboard.py
```

#### カスタムポートで起動
```bash
streamlit run frontend/simple_dashboard.py --server.port 8502
```

#### WSL環境での起動
```bash
# 外部からアクセス可能にする
streamlit run frontend/simple_dashboard.py --server.address 0.0.0.0
```

#### デバッグモードで起動
```bash
streamlit run frontend/simple_dashboard.py --logger.level debug
```

## 📝 起動チェックリスト

1. **仮想環境の確認**
   ```bash
   which python  # 仮想環境のPythonパスが表示されることを確認
   ```

2. **依存関係の確認**
   ```bash
   pip list | grep -E "fastapi|streamlit|pandas"
   ```

3. **ポートの確認**
   ```bash
   # Linux/Mac
   lsof -i :8000
   lsof -i :8501
   
   # Windows
   netstat -an | findstr :8000
   netstat -an | findstr :8501
   ```

## 🔄 システムの再起動

### Core APIの再起動
1. ターミナルで `Ctrl+C` を押してAPIを停止
2. 再度 `python tunnel_core_api.py` を実行

### ダッシュボードの再起動
1. ターミナルで `Ctrl+C` を押してStreamlitを停止
2. 再度 `streamlit run frontend/simple_dashboard.py` を実行

## 🔌 API接続の確認

1. ダッシュボードにアクセス（http://localhost:8501）
2. 左側のサイドバーで「API URL」を確認
   - デフォルト: `http://localhost:8000`
   - カスタムポート使用時: `http://localhost:8001`
3. 「🔄 API接続確認」ボタンをクリック
4. 「✅ API接続: HEALTHY」と表示されれば成功

## ⚠️ トラブルシューティング

### "Module not found" エラー
```bash
# 仮想環境が有効化されているか確認
source .venv/bin/activate

# 必要なパッケージをインストール
pip install -r core/requirements.txt
pip install -r frontend/requirements.txt
```

### ポートが使用中
```bash
# 別のポートで起動
# API
uvicorn tunnel_core_api:app --port 8001

# ダッシュボード
streamlit run frontend/simple_dashboard.py --server.port 8502
```

### API接続エラー
1. Core APIが起動しているか確認
2. API URLが正しいか確認（ポート番号を含む）
3. ファイアウォールがブロックしていないか確認

### モデルファイルの警告
```
WARNING - Model file not found at models/lightgbm_model_optimized.pkl
```
これは正常な警告です。モデルファイルがある場合は以下に配置：
```bash
# 例: モデルファイルがデスクトップにある場合
cp /mnt/c/users/kishida/Desktop/lightgbm_model_optimized.pkl /mnt/c/users/kishida/cursorproject/k-tes_v2_project/core/models/

# または、現在のディレクトリから相対パスで
cd /mnt/c/users/kishida/cursorproject/k-tes_v2_project
cp ../path/to/lightgbm_model_optimized.pkl core/models/
```

## 📊 使用例

### 基本的な使用フロー
1. 両サービスを起動
2. ダッシュボードにアクセス
3. API接続を確認
4. Excelファイルをアップロード
5. パラメータを設定
6. 予測を実行（モデルファイルが必要）

### 開発時の推奨セットアップ
```bash
# ターミナル1: Core API（自動リロード有効）
cd core && uvicorn tunnel_core_api:app --reload

# ターミナル2: Streamlit
streamlit run frontend/simple_dashboard.py

# ターミナル3: 開発作業用
# コード編集、Git操作など
```

## 🛑 システムの停止

1. 各ターミナルで `Ctrl+C` を押してサービスを停止
2. 仮想環境を無効化する場合: `deactivate`

---

最終更新日: 2025年1月16日