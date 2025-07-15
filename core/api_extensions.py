"""
Core APIの拡張機能（学習・データ管理）
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_trainer import TunnelModelTrainer
from ml.data_manager import TrainingDataManager
import json
import asyncio
from datetime import datetime
import uuid
import io
import base64

# ルーター作成
router = APIRouter(prefix="/ml", tags=["machine_learning"])

# グローバルインスタンス
trainer = TunnelModelTrainer()
data_manager = TrainingDataManager(data_dir="../data")  # 相対パスを修正

# ==================== データモデル ====================
class TrainingConfig(BaseModel):
    """学習設定"""
    cv_folds: int = Field(default=5, description="クロスバリデーションの分割数")
    test_size: float = Field(default=0.2, description="テストデータの割合")
    learning_rate: float = Field(default=0.05, description="学習率")
    num_leaves: int = Field(default=31, description="葉の数")
    feature_fraction: float = Field(default=0.9, description="特徴量のサンプリング率")
    bagging_fraction: float = Field(default=0.8, description="データのサンプリング率")
    include_original: bool = Field(default=True, description="元の学習データを含める")
    selected_features: Optional[List[str]] = Field(default=None, description="選択された特徴量のリスト")

class SavePredictionRequest(BaseModel):
    """予測結果の保存リクエスト"""
    prediction_id: str
    actual_pattern: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class TrainingProgress(BaseModel):
    """学習進捗"""
    status: str
    current_step: str
    progress: float
    message: str
    results: Optional[Dict[str, Any]] = None

# ==================== 学習状態管理 ====================
training_status = {
    "is_training": False,
    "progress": 0,
    "current_step": "",
    "message": "",
    "results": None
}

# ==================== エンドポイント ====================
@router.get("/training/status")
async def get_training_status() -> TrainingProgress:
    """学習状態の取得"""
    return TrainingProgress(
        status="training" if training_status["is_training"] else "idle",
        current_step=training_status["current_step"],
        progress=training_status["progress"],
        message=training_status["message"],
        results=training_status["results"]
    )

@router.get("/data/summary")
async def get_data_summary():
    """学習データの概要を取得"""
    try:
        summary = data_manager.get_training_data_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/save_prediction")
async def save_prediction_data(
    prediction_id: str = Form(...),
    actual_pattern: str = Form(...),
    confidence: float = Form(...),
    input_data: str = Form(...),  # JSON string
    metadata: Optional[str] = Form(None)  # JSON string
):
    """予測結果を学習データとして保存"""
    try:
        # JSONデータをパース
        input_df = pd.DataFrame([json.loads(input_data)])
        metadata_dict = json.loads(metadata) if metadata else None
        
        # データを保存
        data_manager.save_prediction_for_training(
            prediction_id=prediction_id,
            input_data=input_df,
            predicted_pattern=input_df.get('predicted_pattern', 'unknown'),
            actual_pattern=actual_pattern,
            confidence=confidence,
            metadata=metadata_dict
        )
        
        return {"status": "success", "message": "データを保存しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_async(config: TrainingConfig):
    """非同期でモデルを学習"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["progress"] = 0
        training_status["current_step"] = "データ準備中"
        training_status["message"] = "学習データを準備しています..."
        
        # データの準備
        training_data = data_manager.prepare_training_data(include_original=config.include_original)
        if training_data.empty:
            raise ValueError("学習データがありません")
        
        training_status["progress"] = 20
        training_status["current_step"] = "特徴量準備中"
        training_status["message"] = f"データ数: {len(training_data)}件"
        
        # 特徴量の準備
        X, y = trainer.prepare_features(training_data, selected_features=config.selected_features)
        
        training_status["progress"] = 40
        training_status["current_step"] = "モデル学習中"
        training_status["message"] = "LightGBMモデルを学習しています..."
        
        # パラメータの設定
        params = {
            'learning_rate': config.learning_rate,
            'num_leaves': config.num_leaves,
            'feature_fraction': config.feature_fraction,
            'bagging_fraction': config.bagging_fraction
        }
        
        # モデルの学習
        model, results = trainer.train_model(
            X, y,
            params=params,
            cv_folds=config.cv_folds,
            test_size=config.test_size,
            show_progress=False
        )
        
        training_status["progress"] = 80
        training_status["current_step"] = "モデル保存中"
        training_status["message"] = "学習済みモデルを保存しています..."
        
        # モデルの保存
        trainer.save_model(model)
        
        # 結果の可視化
        fig = trainer.plot_training_results(results)
        
        # 画像をBase64エンコード
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        training_status["progress"] = 100
        training_status["current_step"] = "完了"
        training_status["message"] = "学習が完了しました"
        training_status["results"] = {
            "accuracy": results["test_accuracy"],
            "cv_score": results["cv_mean_score"],
            "cv_std": results["cv_std_score"],
            "visualization": img_base64
        }
        
    except Exception as e:
        training_status["current_step"] = "エラー"
        training_status["message"] = f"エラーが発生しました: {str(e)}"
        training_status["results"] = None
    finally:
        training_status["is_training"] = False

@router.post("/train")
async def start_training(config: TrainingConfig):
    """モデルの学習を開始"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="既に学習が実行中です")
    
    # 非同期で学習を開始
    asyncio.create_task(train_model_async(config))
    
    return {
        "status": "started",
        "message": "学習を開始しました"
    }


@router.post("/train/custom")
async def start_training_custom(
    training_file: UploadFile = File(...),
    cv_folds: int = Form(5),
    test_size: float = Form(0.2),
    learning_rate: float = Form(0.05),
    num_leaves: int = Form(31),
    feature_fraction: float = Form(0.9),
    bagging_fraction: float = Form(0.8),
    include_original: bool = Form(True),
    selected_features: Optional[str] = Form(None)  # JSON文字列として受け取る
):
    """カスタムファイルでモデルの学習を開始"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="既に学習が実行中です")
    
    try:
        # ファイルを読み込み
        contents = await training_file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding='shift-jis')
        
        # 一時的にカスタムデータを保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='shift-jis') as tmp:
            df.to_csv(tmp, index=False)
            tmp_path = tmp.name
        
        # 選択された特徴量をパース
        selected_features_list = None
        if selected_features:
            try:
                selected_features_list = json.loads(selected_features)
            except:
                selected_features_list = None
        
        # カスタム学習設定
        config = TrainingConfig(
            cv_folds=cv_folds,
            test_size=test_size,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            include_original=include_original,
            selected_features=selected_features_list
        )
        
        # カスタムファイルパスを含めた非同期学習
        async def train_with_custom_file():
            try:
                # カスタムファイルを読み込み
                custom_data = pd.read_csv(tmp_path, encoding='shift-jis')
                
                # 既存データと結合（include_originalがTrueの場合）
                if config.include_original:
                    original_data = data_manager.prepare_training_data(include_original=True)
                    if not original_data.empty:
                        # カスタムデータと既存データを結合
                        training_data = pd.concat([original_data, custom_data], ignore_index=True)
                    else:
                        training_data = custom_data
                else:
                    training_data = custom_data
                
                # データが空でないことを確認
                if training_data.empty:
                    raise ValueError("学習データが空です")
                
                # 通常の学習プロセスを実行
                global training_status
                training_status["is_training"] = True
                training_status["progress"] = 0
                training_status["current_step"] = "データ準備中"
                training_status["message"] = f"カスタムファイル（{training_file.filename}）を使用して学習しています..."
                
                # 特徴量の準備
                X, y = trainer.prepare_features(training_data, selected_features=config.selected_features)
                
                training_status["progress"] = 40
                training_status["current_step"] = "モデル学習中"
                training_status["message"] = f"データ数: {len(training_data)}件（カスタム: {len(custom_data)}件）"
                
                # パラメータの設定
                params = {
                    'learning_rate': config.learning_rate,
                    'num_leaves': config.num_leaves,
                    'feature_fraction': config.feature_fraction,
                    'bagging_fraction': config.bagging_fraction
                }
                
                # モデルの学習
                model, results = trainer.train_model(
                    X, y,
                    params=params,
                    cv_folds=config.cv_folds,
                    test_size=config.test_size,
                    show_progress=False
                )
                
                training_status["progress"] = 80
                training_status["current_step"] = "モデル保存中"
                training_status["message"] = "学習済みモデルを保存しています..."
                
                # モデルの保存
                trainer.save_model(model)
                
                # 結果の可視化
                fig = trainer.plot_training_results(results)
                
                # 画像をBase64エンコード
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close(fig)
                
                training_status["progress"] = 100
                training_status["current_step"] = "完了"
                training_status["message"] = "学習が完了しました"
                training_status["results"] = {
                    "accuracy": results["test_accuracy"],
                    "cv_score": results["cv_mean_score"],
                    "cv_std": results["cv_std_score"],
                    "visualization": img_base64,
                    "custom_file": training_file.filename,
                    "custom_data_count": len(custom_data)
                }
                
            except Exception as e:
                training_status["current_step"] = "エラー"
                training_status["message"] = f"エラーが発生しました: {str(e)}"
                training_status["results"] = None
            finally:
                training_status["is_training"] = False
                # 一時ファイルを削除
                import os
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        # 非同期で学習を開始
        asyncio.create_task(train_with_custom_file())
        
        return {
            "status": "started",
            "message": f"カスタムファイル（{training_file.filename}）での学習を開始しました"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイル読み込みエラー: {str(e)}")

@router.post("/data/clear")
async def clear_accumulated_data(backup: bool = True):
    """蓄積データをクリア"""
    try:
        data_manager.clear_accumulated_data(backup=backup)
        return {"status": "success", "message": "蓄積データをクリアしました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/exists")
async def check_model_exists():
    """モデルの存在確認"""
    from pathlib import Path
    model_path = Path("core/models/lightgbm_model_optimized.pkl")
    return {"exists": model_path.exists()}


@router.get("/data/download/{filename}")
async def download_training_file(filename: str):
    """学習データファイルのダウンロード"""
    from fastapi.responses import FileResponse
    import os
    
    file_path = Path("../data/training") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # セキュリティチェック（ディレクトリトラバーサル防止）
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='text/csv'
    )


@router.delete("/data/file/{filename}")
async def delete_training_file(filename: str):
    """学習データファイルの削除"""
    file_path = Path("../data/training") / filename
    
    # セキュリティチェック
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return {"status": "success", "message": f"{filename} を削除しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/upload")
async def upload_training_file(file: UploadFile = File(...)):
    """学習データファイルのアップロード"""
    # ファイル名の検証
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # セキュリティチェック
    safe_filename = Path(file.filename).name
    if ".." in safe_filename or "/" in safe_filename or "\\" in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # 保存先パス
    file_path = Path("../data/training") / safe_filename
    
    # 既存ファイルのチェック
    if file_path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    
    try:
        # ファイルを保存
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        return {"status": "success", "message": f"{safe_filename} をアップロードしました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# matplotlib のインポート（必要な場合）
try:
    import matplotlib
    matplotlib.use('Agg')  # GUIなし環境用
    import matplotlib.pyplot as plt
except ImportError:
    plt = None