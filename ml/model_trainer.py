"""
LightGBMモデルの学習機能
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class TunnelModelTrainer:
    """トンネル支保パターン予測モデルの学習クラス"""
    
    def __init__(self, model_dir: str = "core/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.training_history = []
        
    def load_training_data(self, csv_path: str, encoding: str = 'shift_jis') -> pd.DataFrame:
        """学習データの読み込み"""
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"データ読み込み完了: {df.shape[0]}行 × {df.shape[1]}列")
        return df
    
    def prepare_features(self, df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量とラベルの準備"""
        # ラベル列
        label_col = '支保パターン'
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}'カラムが見つかりません")
        
        # 特徴量列の検出
        if selected_features is not None:
            # ユーザーが選択した特徴量を使用
            feature_cols = []
            for feature in selected_features:
                if feature in df.columns:
                    feature_cols.append(feature)
                else:
                    print(f"警告: 選択された特徴量 '{feature}' がデータに存在しません")
            
            if len(feature_cols) == 0:
                raise ValueError("選択された特徴量がデータに存在しません")
                
            print(f"ユーザー選択の特徴量を使用: {len(feature_cols)}個")
        else:
            # デフォルトの特徴量検出
            feature_cols = []
            
            # パターン1: 正確な形式の特徴量名を探す
            for param in ['回転圧[MPa]', '打撃圧[MPa]', 'フィード圧[MPa]', 
                         '穿孔速度[cm/秒]', '穿孔エネルギー[J/cm^3]']:
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    col_name = f"{param}_{stat}"
                    if col_name in df.columns:
                        feature_cols.append(col_name)
            
            # パターン2: 統計量を含むカラムを探す（カラム名が異なる場合）
            if len(feature_cols) == 0:
                print("標準的な特徴量名が見つかりません。統計量を含むカラムを探します...")
                for col in df.columns:
                    if col != label_col and any(stat in str(col) for stat in ['mean', 'std', 'min', 'max', '25%', '50%', '75%']):
                        feature_cols.append(col)
            
            # 区間距離も特徴量に追加
            if '区間距離' in df.columns:
                feature_cols.append('区間距離')
            
            # パターン3: 数値カラムをすべて使用（最終手段）
            if len(feature_cols) == 0:
                print("統計量カラムが見つかりません。すべての数値カラムを使用します...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != label_col]
            
            if len(feature_cols) == 0:
                raise ValueError("使用可能な特徴量カラムが見つかりません")
        
        self.feature_columns = feature_cols
        
        # 特徴量とラベルの分離
        X = df[feature_cols]
        y = df[label_col]
        
        # ラベルエンコーディング
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"特徴量数: {len(feature_cols)}")
        print(f"クラス数: {len(self.label_encoder.classes_)}")
        print(f"クラス: {list(self.label_encoder.classes_)}")
        
        return X, pd.Series(y_encoded, index=y.index)
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        show_progress: bool = True
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """LightGBMモデルの学習"""
        
        # デフォルトパラメータ
        default_params = {
            'objective': 'multiclass',
            'num_class': len(self.label_encoder.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': random_state
        }
        
        if params:
            default_params.update(params)
        
        # 訓練・テストデータの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # クロスバリデーション
        cv_scores = []
        cv_models = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # 学習履歴の記録用
        eval_results = {'training': {'multi_logloss': []}, 'validation': {'multi_logloss': []}}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # LightGBMデータセットの作成
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            # モデルの学習
            model = lgb.train(
                default_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['training', 'validation'],
                num_boost_round=100,
                callbacks=[
                    lgb.early_stopping(10),
                    lgb.log_evaluation(10 if show_progress else 0)
                ]
            )
            
            # 予測と評価
            y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_fold_val, y_pred_class)
            cv_scores.append(accuracy)
            cv_models.append(model)
            
            if show_progress:
                print(f"Fold {fold + 1}/{cv_folds} - Accuracy: {accuracy:.4f}")
        
        # 最良モデルの選択
        best_fold = np.argmax(cv_scores)
        best_model = cv_models[best_fold]
        
        # テストデータでの評価
        y_test_pred = best_model.predict(X_test, num_iteration=best_model.best_iteration)
        y_test_pred_class = np.argmax(y_test_pred, axis=1)
        test_accuracy = accuracy_score(y_test, y_test_pred_class)
        
        # 結果の集計
        results = {
            'cv_scores': cv_scores,
            'cv_mean_score': np.mean(cv_scores),
            'cv_std_score': np.std(cv_scores),
            'test_accuracy': test_accuracy,
            'best_iteration': best_model.best_iteration,
            'feature_importance': dict(zip(
                self.feature_columns,
                best_model.feature_importance(importance_type='gain')
            )),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred_class).tolist(),
            'classification_report': classification_report(
                y_test, y_test_pred_class,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        # 学習履歴に追加
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'params': default_params,
            'results': results
        })
        
        return best_model, results
    
    def save_model(self, model: lgb.Booster, filename: str = "lightgbm_model_optimized.pkl"):
        """モデルの保存"""
        model_path = self.model_dir / filename
        
        # モデルと関連情報を保存
        model_data = {
            'model': model,
            'feature_names': self.feature_columns,
            'label_encoder': self.label_encoder,
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'training_history': self.training_history[-1] if self.training_history else None
        }
        
        joblib.dump(model_data, model_path)
        print(f"モデルを保存しました: {model_path}")
        
        # 学習履歴もJSON形式で保存
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
    
    def plot_training_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """学習結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. CV scores
        ax = axes[0, 0]
        cv_scores = results['cv_scores']
        ax.bar(range(1, len(cv_scores) + 1), cv_scores)
        ax.axhline(y=results['cv_mean_score'], color='r', linestyle='--', 
                  label=f"Mean: {results['cv_mean_score']:.4f}")
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-validation Scores')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # 2. Feature importance (top 10)
        ax = axes[0, 1]
        importance_df = pd.DataFrame(
            list(results['feature_importance'].items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False).head(10)
        
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.invert_yaxis()
        
        # 3. Confusion matrix
        ax = axes[1, 0]
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # 4. Classification report
        ax = axes[1, 1]
        ax.axis('off')
        report_text = f"Test Accuracy: {results['test_accuracy']:.4f}\n\n"
        report_text += f"CV Mean Score: {results['cv_mean_score']:.4f} ± {results['cv_std_score']:.4f}\n\n"
        report_text += "Per-class Performance:\n"
        
        for class_name in self.label_encoder.classes_:
            metrics = results['classification_report'][class_name]
            report_text += f"\n{class_name}:\n"
            report_text += f"  Precision: {metrics['precision']:.3f}\n"
            report_text += f"  Recall: {metrics['recall']:.3f}\n"
            report_text += f"  F1-score: {metrics['f1-score']:.3f}\n"
        
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Classification Report')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"学習結果を保存しました: {save_path}")
        
        return fig


# 使用例
if __name__ == "__main__":
    # トレーナーの初期化
    trainer = TunnelModelTrainer()
    
    # データの読み込み
    df = trainer.load_training_data("data/training/川合TN_AGF_ver3_train_SJIS_20250716_020351.csv")
    
    # 特徴量の準備
    X, y = trainer.prepare_features(df)
    
    # モデルの学習
    model, results = trainer.train_model(X, y, show_progress=True)
    
    # モデルの保存
    trainer.save_model(model)
    
    # 結果の可視化
    trainer.plot_training_results(results, save_path="ml/training_results.png")