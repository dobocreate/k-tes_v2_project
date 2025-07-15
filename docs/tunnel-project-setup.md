# トンネル地山評価システム - プロジェクト構成

## ディレクトリ構造

```
tunnel-evaluation-system/
├── core/                       # コアAPI
│   ├── tunnel_core_api.py     # FastAPI実装
│   ├── requirements.txt        # API依存関係
│   └── models/                # 学習済みモデル
│       └── lightgbm_model_optimized.pkl
│
├── frontend/                   # フロントエンド
│   ├── simple_dashboard.py    # Streamlitアプリ
│   └── requirements.txt       # UI依存関係
│
├── ml/                        # 機械学習開発
│   ├── data_preprocessing.py  # データ前処理
│   ├── feature_engineering.py # 特徴量エンジニアリング
│   ├── model_training.py      # モデル学習
│   ├── optuna_optimization.py # ハイパーパラメータ最適化
│   └── outlier_handler.py     # 外れ値処理
│
├── notebooks/                 # 実験・分析用
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── data/                      # データディレクトリ
│   ├── raw/                  # 生データ
│   ├── processed/            # 前処理済みデータ
│   └── results/              # 実験結果
│
├── tests/                     # テストコード
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── docker/                    # Docker設定
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── docs/                      # ドキュメント
│   ├── API_SPEC.md
│   ├── USER_GUIDE.md
│   └── DEVELOPMENT.md
│
├── .env.example              # 環境変数サンプル
├── README.md
└── setup.py
```

## 環境構築手順

### 1. 基本環境の準備

```bash
# プロジェクトディレクトリ作成
mkdir tunnel-evaluation-system
cd tunnel-evaluation-system

# Python仮想環境作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Core API用の依存関係

**core/requirements.txt**
```
# Core API Dependencies
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# ML Dependencies
lightgbm==4.1.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
openpyxl==3.1.2
joblib==1.3.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Logging and Config
python-dotenv==1.0.0
pydantic==2.5.2
```

### 3. Frontend用の依存関係

**frontend/requirements.txt**
```
# Frontend Dependencies
streamlit==1.29.0
plotly==5.18.0
requests==2.31.0
pandas==2.1.4
openpyxl==3.1.2
```

### 4. 開発環境用の追加依存関係

**requirements-dev.txt**
```
# Development Dependencies
jupyter==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.12.0
flake8==6.1.0
mypy==1.7.1
optuna==3.4.0
```

### 5. インストール手順

```bash
# Core APIのセットアップ
cd core
pip install -r requirements.txt

# Frontendのセットアップ
cd ../frontend
pip install -r requirements.txt

# 開発環境のセットアップ（オプション）
cd ..
pip install -r requirements-dev.txt
```

## 起動手順

### 開発環境での起動

#### 1. Core APIの起動
```bash
cd core
python tunnel_core_api.py
# または
uvicorn tunnel_core_api:app --reload --port 8000
```

#### 2. Frontendの起動（別ターミナル）
```bash
cd frontend
streamlit run simple_dashboard.py
```

#### 3. アクセス
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

### Dockerを使用した起動

**docker/docker-compose.yml**
```yaml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
    environment:
      - MODEL_PATH=/app/models/lightgbm_model_optimized.pkl
    
  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

**docker/Dockerfile.api**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 依存関係のインストール
COPY core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコピー
COPY core/tunnel_core_api.py .

# ポート公開
EXPOSE 8000

# 起動
CMD ["uvicorn", "tunnel_core_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker/Dockerfile.frontend**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 依存関係のインストール
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコピー
COPY frontend/simple_dashboard.py .

# ポート公開
EXPOSE 8501

# 起動
CMD ["streamlit", "run", "simple_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Docker Composeで起動
cd docker
docker-compose up --build
```

## 環境変数設定

**.env.example**
```env
# API設定
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/lightgbm_model_optimized.pkl

# Frontend設定
STREAMLIT_SERVER_PORT=8501
API_BASE_URL=http://localhost:8000

# ログ設定
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 開発フロー

### 1. 機械学習モデルの開発
```bash
cd ml
# データ前処理
python data_preprocessing.py

# モデル学習
python model_training.py

# 最適化
python optuna_optimization.py
```

### 2. APIのテスト
```bash
cd tests
pytest test_api.py -v
```

### 3. 統合テスト
```python
# test_integration.py
import requests

# APIヘルスチェック
response = requests.get("http://localhost:8000/")
assert response.status_code == 200

# 予測テスト
with open("test_data.xlsx", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files,
        data={
            "tunnel_name": "test",
            "previous_support_pattern": "CI"
        }
    )
    assert response.status_code == 200
```

## 次のステップ

### Phase 1: コア機能の完成（現在）
- [x] 前処理機能
- [x] 予測API
- [x] 簡易UI

### Phase 2: 機能拡張
- [ ] バッチ処理対応
- [ ] 予測履歴のDB保存
- [ ] ユーザー認証

### Phase 3: 本格的なダッシュボード
- [ ] リアルタイムモニタリング
- [ ] 複数トンネルの管理
- [ ] レポート生成機能
- [ ] 管理者機能

### Phase 4: 運用最適化
- [ ] 自動再学習
- [ ] A/Bテスト機能
- [ ] 性能モニタリング

## トラブルシューティング

### よくある問題と解決策

1. **モデルファイルが見つからない**
   ```bash
   # モデルディレクトリを作成
   mkdir -p core/models
   # 学習済みモデルを配置
   cp path/to/lightgbm_model_optimized.pkl core/models/
   ```

2. **ポートが使用中**
   ```bash
   # 使用中のポートを確認
   lsof -i :8000
   lsof -i :8501
   # 別のポートを指定
   uvicorn tunnel_core_api:app --port 8001
   ```

3. **依存関係のエラー**
   ```bash
   # 仮想環境を再作成
   deactivate
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## リソース

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)