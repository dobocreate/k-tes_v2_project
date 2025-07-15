# トンネル地山評価システム

穿孔データから機械学習により支保パターンを予測するシステムです。

## 🚀 概要

本システムは、トンネル掘削時の穿孔データを分析し、適切な支保パターンを予測します。

### 主な機能
- 穿孔データの前処理（外れ値除去、統計量計算）
- LightGBMモデルによる支保パターン予測
- 予測結果の可視化
- Web UIによる簡単な操作

### システム構成
- **Core API**: FastAPIによる予測エンドポイント
- **Frontend**: Streamlitによるダッシュボード
- **ML Model**: LightGBMによる機械学習モデル

## 📋 必要条件

- Python 3.9以上
- pip（Pythonパッケージマネージャー）

## 🛠️ インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd k-tes_v2_project
```

### 2. Python仮想環境の作成

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. 依存関係のインストール

```bash
# Core APIの依存関係
cd core
pip install -r requirements.txt

# Frontendの依存関係
cd ../frontend
pip install -r requirements.txt
```

### 4. 環境変数の設定

```bash
# プロジェクトルートに戻る
cd ..

# 環境変数ファイルをコピー
cp .env.example .env

# 必要に応じて.envファイルを編集
```

### 5. モデルファイルの配置

学習済みモデルファイル（`lightgbm_model_optimized.pkl`）を`core/models/`ディレクトリに配置してください。

```bash
# modelsディレクトリが存在しない場合は作成
mkdir -p core/models

# モデルファイルをコピー
cp path/to/lightgbm_model_optimized.pkl core/models/
```

## 🚀 起動方法

### 1. Core APIの起動

```bash
# 新しいターミナルで
cd core
python tunnel_core_api.py
```

APIは`http://localhost:8000`で起動します。

### 2. Frontendの起動

```bash
# 別の新しいターミナルで
cd frontend
streamlit run simple_dashboard.py
```

ダッシュボードは`http://localhost:8501`で起動します。

### 3. アクセス

ブラウザで以下にアクセス：
- **ダッシュボード**: http://localhost:8501
- **API ドキュメント**: http://localhost:8000/docs

## 📊 使い方

### データ準備

Excelファイル（.xlsx）に以下のカラムを含む穿孔データを準備：

| カラム名 | 説明 | 単位 |
|---------|------|------|
| 測定位置 | 測定した位置 | m |
| 回転圧 | ドリルの回転圧力 | MPa |
| 打撃圧 | ドリルの打撃圧力 | MPa |
| フィード圧 | ドリルのフィード圧力 | MPa |
| 穿孔速度 | 穿孔の速度 | mm/min |
| 穿孔エネルギー | 穿孔に必要なエネルギー | J |

### 予測実行

1. ダッシュボードにアクセス
2. サイドバーでAPI接続を確認
3. データファイルをアップロード
4. トンネル名と前回の支保パターンを入力
5. 「支保パターンを予測」ボタンをクリック

### 支保パターン

- **CII-b**: 最も軽い支保（安定地山）
- **CII**: 軽い支保
- **CI**: 標準的な支保
- **DI**: やや重い支保
- **DIIa**: 重い支保
- **DIIa-Au**: 重い支保＋補助工法
- **DIIIa-Au**: 最も重い支保＋補助工法

## 🔧 設定

### 前処理パラメータ

- **窓サイズ**: 統計量を計算する区間のデータ点数（デフォルト: 10）
- **外れ値除去**: IQR法による外れ値除去の有効/無効

### API設定

`.env`ファイルで以下を設定可能：

```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/lightgbm_model_optimized.pkl
```

## 📁 プロジェクト構造

```
k-tes_v2_project/
├── core/                      # Core API
│   ├── tunnel_core_api.py    # FastAPI実装
│   ├── requirements.txt       # API依存関係
│   └── models/               # 学習済みモデル
│
├── frontend/                  # フロントエンド
│   ├── simple_dashboard.py   # Streamlitアプリ
│   └── requirements.txt      # UI依存関係
│
├── ml/                       # 機械学習開発（将来拡張用）
├── data/                     # データディレクトリ
├── tests/                    # テストコード（将来拡張用）
├── docker/                   # Docker設定（将来拡張用）
│
├── .env.example              # 環境変数サンプル
└── README.md                 # このファイル
```

## 🐛 トラブルシューティング

### モデルファイルが見つからない

```bash
# modelsディレクトリを作成
mkdir -p core/models

# モデルファイルを配置
cp path/to/lightgbm_model_optimized.pkl core/models/
```

### ポートが使用中

```bash
# 使用中のポートを確認（Linux/Mac）
lsof -i :8000
lsof -i :8501

# 別のポートを使用
# .envファイルで設定を変更
API_PORT=8001
STREAMLIT_SERVER_PORT=8502
```

### 依存関係のエラー

```bash
# 仮想環境を再作成
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係を再インストール
pip install -r core/requirements.txt
pip install -r frontend/requirements.txt
```

## 📝 API仕様

### エンドポイント

#### GET /
ヘルスチェック

#### POST /predict
支保パターン予測

**パラメータ:**
- `file`: 穿孔データのExcelファイル
- `tunnel_name`: トンネル名
- `previous_support_pattern`: 前回の支保パターン
- `window_size`: 統計量計算の窓サイズ
- `remove_outliers`: 外れ値除去の有無

**レスポンス:**
```json
{
  "prediction_id": "pred_20241101_123456",
  "timestamp": "2024-11-01T12:34:56",
  "tunnel_name": "新トンネル",
  "predicted_pattern": "CI",
  "probability_distribution": {
    "CII-b": 0.05,
    "CII": 0.10,
    "CI": 0.70,
    "DI": 0.15
  },
  "confidence_score": 0.70,
  "preprocessing_stats": {
    "original_rows": 1000,
    "outliers_removed": 10,
    "sections_created": 99
  }
}
```

## 🤝 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📄 ライセンス

[ライセンスを記載]

## 👥 開発者

[開発者情報を記載]

## 📞 サポート

質問や問題がある場合は、[連絡先]までお問い合わせください。