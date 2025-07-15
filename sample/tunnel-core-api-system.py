# tunnel_core_api.py
"""
トンネル地山評価システム - コアAPI
前処理、推論、結果返却を担当するマイクロサービス
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APIインスタンス
app = FastAPI(
    title="トンネル地山評価 Core API",
    description="穿孔データの前処理と支保パターン予測を行うAPI",
    version="0.1.0"
)

# ==================== データモデル定義 ====================
class DrillingDataPoint(BaseModel):
    """穿孔データの1点"""
    position: float = Field(..., description="測定位置 (m)")
    rotation_pressure: float = Field(..., description="回転圧 (MPa)")
    impact_pressure: float = Field(..., description="打撃圧 (MPa)")
    feed_pressure: float = Field(..., description="フィード圧 (MPa)")
    drilling_speed: float = Field(..., description="穿孔速度 (mm/min)")
    drilling_energy: float = Field(..., description="穿孔エネルギー (J)")

class PreprocessingConfig(BaseModel):
    """前処理の設定"""
    window_size: int = Field(default=10, description="統計量計算の窓サイズ")
    remove_outliers: bool = Field(default=True, description="外れ値除去の有無")
    outlier_method: str = Field(default="IQR_3.0", description="外れ値除去方法")
    target_features: List[str] = Field(
        default=["穿孔エネルギー"], 
        description="外れ値除去対象の特徴量"
    )

class PredictionRequest(BaseModel):
    """予測リクエスト"""
    tunnel_name: str = Field(..., description="トンネル名")
    previous_support_pattern: str = Field(..., description="前回の支保パターン")
    preprocessing_config: Optional[PreprocessingConfig] = Field(
        default=PreprocessingConfig(),
        description="前処理設定"
    )

class PredictionResponse(BaseModel):
    """予測レスポンス"""
    prediction_id: str
    timestamp: str
    tunnel_name: str
    predicted_pattern: str
    probability_distribution: Dict[str, float]
    confidence_score: float
    preprocessing_stats: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    model_loaded: bool
    model_version: str
    last_updated: Optional[str]

# ==================== コア処理クラス ====================
class TunnelCoreProcessor:
    """コア処理を担当するクラス"""
    
    def __init__(self, model_path: str = "models/lightgbm_model_optimized.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = None
        self.feature_names = None
        self.label_encoder = None
        self.load_model()
        
    def load_model(self):
        """学習済みモデルの読み込み"""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.label_encoder = model_data.get('label_encoder')
                self.model_info = {
                    'version': model_data.get('version', 'unknown'),
                    'last_updated': model_data.get('last_updated', 'unknown')
                }
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_drilling_data(
        self, 
        df: pd.DataFrame, 
        config: PreprocessingConfig
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        穿孔データの前処理
        
        Returns:
            - features_df: 特徴量DataFrame
            - preprocessing_stats: 前処理の統計情報
        """
        stats = {
            'original_rows': len(df),
            'outliers_removed': 0,
            'sections_created': 0
        }
        
        # 1. 外れ値除去（設定に応じて）
        if config.remove_outliers:
            df_cleaned, outliers_removed = self._remove_outliers(
                df, 
                config.target_features,
                config.outlier_method
            )
            stats['outliers_removed'] = outliers_removed
        else:
            df_cleaned = df
            
        # 2. 区間ごとの統計量計算
        features_list = []
        base_features = ['回転圧', '打撃圧', 'フィード圧', '穿孔速度', '穿孔エネルギー']
        
        for i in range(0, len(df_cleaned), config.window_size):
            section = df_cleaned.iloc[i:i+config.window_size]
            
            if len(section) < config.window_size * 0.8:
                continue
                
            section_features = {}
            
            # 各特徴量の統計量
            for feature in base_features:
                if feature in section.columns:
                    values = section[feature].dropna()
                    if len(values) > 0:
                        section_features[f'{feature}_mean'] = values.mean()
                        section_features[f'{feature}_std'] = values.std()
                        section_features[f'{feature}_min'] = values.min()
                        section_features[f'{feature}_max'] = values.max()
                        section_features[f'{feature}_q25'] = values.quantile(0.25)
                        section_features[f'{feature}_q50'] = values.quantile(0.50)
                        section_features[f'{feature}_q75'] = values.quantile(0.75)
            
            # 区間距離
            section_features['section_distance'] = (
                section['測定位置'].iloc[-1] - section['測定位置'].iloc[0]
            )
            
            features_list.append(section_features)
        
        features_df = pd.DataFrame(features_list)
        stats['sections_created'] = len(features_df)
        stats['processed_rows'] = len(df_cleaned)
        
        return features_df, stats
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame, 
        target_features: List[str],
        method: str
    ) -> tuple[pd.DataFrame, int]:
        """外れ値除去の実装"""
        df_cleaned = df.copy()
        all_outliers = set()
        
        for feature in target_features:
            if feature not in df.columns:
                continue
                
            if method == "IQR_3.0":
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3.0 * IQR
                upper = Q3 + 3.0 * IQR
                outliers = df[(df[feature] < lower) | (df[feature] > upper)].index
                all_outliers.update(outliers)
                
        df_cleaned = df.drop(index=list(all_outliers))
        return df_cleaned, len(all_outliers)
    
    def predict(
        self, 
        features_df: pd.DataFrame,
        previous_support_pattern: str,
        tunnel_name: str
    ) -> PredictionResponse:
        """支保パターンの予測"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # 前支保パターンの追加
        features_df['previous_support_pattern'] = previous_support_pattern
        features_df['tunnel_name'] = tunnel_name
        
        # 必要な特徴量の選択（モデルの学習時と同じ順序）
        X = features_df[self.feature_names]
        
        # 予測
        y_proba = self.model.predict(X)
        y_pred_idx = np.argmax(y_proba, axis=1)
        
        # 最も確率の高い予測を取得
        if len(y_pred_idx) > 0:
            # 複数セクションの場合は最頻値
            predicted_idx = np.bincount(y_pred_idx).argmax()
            confidence = y_proba[:, predicted_idx].mean()
        else:
            predicted_idx = y_pred_idx[0]
            confidence = y_proba[0, predicted_idx]
        
        # ラベルデコード
        if self.label_encoder:
            predicted_pattern = self.label_encoder.inverse_transform([predicted_idx])[0]
            prob_dist = {
                self.label_encoder.inverse_transform([i])[0]: float(y_proba[:, i].mean())
                for i in range(len(self.label_encoder.classes_))
            }
        else:
            predicted_pattern = str(predicted_idx)
            prob_dist = {str(i): float(y_proba[:, i].mean()) for i in range(y_proba.shape[1])}
        
        # レスポンス作成
        response = PredictionResponse(
            prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            tunnel_name=tunnel_name,
            predicted_pattern=predicted_pattern,
            probability_distribution=prob_dist,
            confidence_score=float(confidence),
            preprocessing_stats={},
            feature_importance=self._get_feature_importance()
        )
        
        return response
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度の取得"""
        if hasattr(self.model, 'feature_importance'):
            importance = self.model.feature_importance(importance_type='gain')
            return {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
                if imp > 0
            }
        return None

# ==================== 可視化ヘルパー ====================
class VisualizationHelper:
    """可視化機能を提供するヘルパークラス"""
    
    @staticmethod
    def create_prediction_plot(response: PredictionResponse) -> str:
        """予測結果の可視化（Base64エンコード画像）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 確率分布の棒グラフ
        patterns = list(response.probability_distribution.keys())
        probs = list(response.probability_distribution.values())
        
        bars = ax1.bar(patterns, probs)
        ax1.set_xlabel('支保パターン')
        ax1.set_ylabel('予測確率')
        ax1.set_title('支保パターンの予測確率分布')
        ax1.set_ylim(0, 1)
        
        # 予測パターンをハイライト
        predicted_idx = patterns.index(response.predicted_pattern)
        bars[predicted_idx].set_color('red')
        
        # 確率値を表示
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom')
        
        # 特徴量重要度（上位10個）
        if response.feature_importance:
            importance_df = pd.DataFrame(
                list(response.feature_importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False).head(10)
            
            ax2.barh(importance_df['feature'], importance_df['importance'])
            ax2.set_xlabel('重要度')
            ax2.set_title('特徴量重要度 (Top 10)')
        
        plt.tight_layout()
        
        # Base64エンコード
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_str

# ==================== グローバルインスタンス ====================
processor = TunnelCoreProcessor()
viz_helper = VisualizationHelper()

# ==================== APIエンドポイント ====================
@app.get("/", response_model=HealthResponse)
async def root():
    """ヘルスチェック"""
    return HealthResponse(
        status="healthy",
        model_loaded=processor.model is not None,
        model_version=processor.model_info.get('version', 'unknown') if processor.model_info else 'unknown',
        last_updated=processor.model_info.get('last_updated') if processor.model_info else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="穿孔データのExcelファイル"),
    tunnel_name: str = "unknown",
    previous_support_pattern: str = "CI",
    window_size: int = 10,
    remove_outliers: bool = True
):
    """
    穿孔データから支保パターンを予測
    """
    try:
        # ファイル読み込み
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # カラム名の正規化（想定されるバリエーションに対応）
        column_mapping = {
            '測定位置': ['測定位置', '位置', 'position', 'Position'],
            '回転圧': ['回転圧', 'rotation_pressure', 'Rotation Pressure'],
            '打撃圧': ['打撃圧', 'impact_pressure', 'Impact Pressure'],
            'フィード圧': ['フィード圧', 'feed_pressure', 'Feed Pressure'],
            '穿孔速度': ['穿孔速度', 'drilling_speed', 'Drilling Speed'],
            '穿孔エネルギー': ['穿孔エネルギー', 'drilling_energy', 'Drilling Energy']
        }
        
        for std_name, variations in column_mapping.items():
            for var in variations:
                if var in df.columns:
                    df.rename(columns={var: std_name}, inplace=True)
                    break
        
        # 前処理設定
        config = PreprocessingConfig(
            window_size=window_size,
            remove_outliers=remove_outliers
        )
        
        # 前処理実行
        features_df, preprocessing_stats = processor.preprocess_drilling_data(df, config)
        
        if len(features_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="前処理後に有効なデータがありません"
            )
        
        # 予測実行
        response = processor.predict(
            features_df,
            previous_support_pattern,
            tunnel_name
        )
        response.preprocessing_stats = preprocessing_stats
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/visualization")
async def predict_with_visualization(
    file: UploadFile = File(...),
    tunnel_name: str = "unknown",
    previous_support_pattern: str = "CI",
    window_size: int = 10,
    remove_outliers: bool = True
):
    """
    予測結果を可視化付きで返す
    """
    # 通常の予測を実行
    response = await predict(
        file=file,
        tunnel_name=tunnel_name,
        previous_support_pattern=previous_support_pattern,
        window_size=window_size,
        remove_outliers=remove_outliers
    )
    
    # 可視化を追加
    plot_base64 = viz_helper.create_prediction_plot(response)
    
    return {
        "prediction": response.dict(),
        "visualization": plot_base64
    }

@app.post("/preprocess")
async def preprocess_only(
    file: UploadFile = File(...),
    window_size: int = 10,
    remove_outliers: bool = True
):
    """
    前処理のみ実行（デバッグ用）
    """
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        config = PreprocessingConfig(
            window_size=window_size,
            remove_outliers=remove_outliers
        )
        
        features_df, stats = processor.preprocess_drilling_data(df, config)
        
        return {
            "preprocessing_stats": stats,
            "features_shape": features_df.shape,
            "features_sample": features_df.head().to_dict(),
            "feature_names": features_df.columns.tolist()
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 簡易クライアント ====================
"""
# client_example.py
import requests
import json

# APIのベースURL
BASE_URL = "http://localhost:8000"

# ヘルスチェック
response = requests.get(f"{BASE_URL}/")
print("Health check:", response.json())

# 予測実行
with open("drilling_data.xlsx", "rb") as f:
    files = {"file": f}
    data = {
        "tunnel_name": "新トンネル",
        "previous_support_pattern": "CII",
        "window_size": 10,
        "remove_outliers": True
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"予測結果: {result['predicted_pattern']}")
        print(f"信頼度: {result['confidence_score']:.2%}")
        print(f"確率分布: {result['probability_distribution']}")
    else:
        print(f"Error: {response.text}")
"""

# ==================== 起動スクリプト ====================
if __name__ == "__main__":
    import uvicorn
    
    # 開発環境での起動
    uvicorn.run(
        "tunnel_core_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )