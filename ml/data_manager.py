"""
学習データと予測データの管理
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import uuid


class TrainingDataManager:
    """学習データの管理クラス"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.prediction_dir = self.data_dir / "predictions"
        self.accumulated_dir = self.data_dir / "accumulated"
        
        # ディレクトリ作成
        for dir_path in [self.training_dir, self.prediction_dir, self.accumulated_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 蓄積データ用CSVのパス
        self.accumulated_csv = self.accumulated_dir / "accumulated_data.csv"
        self.metadata_json = self.accumulated_dir / "metadata.json"
        
    def save_prediction_for_training(
        self,
        prediction_id: str,
        input_data: pd.DataFrame,
        predicted_pattern: str,
        actual_pattern: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """予測データを学習用に保存"""
        
        # 既存の蓄積データを読み込み
        if self.accumulated_csv.exists():
            accumulated_df = pd.read_csv(self.accumulated_csv, encoding='utf-8')
        else:
            accumulated_df = pd.DataFrame()
        
        # 新しいデータを追加
        new_row = input_data.copy()
        new_row['支保パターン'] = actual_pattern  # 実際のパターンを正解ラベルとして保存
        new_row['予測パターン'] = predicted_pattern
        new_row['予測信頼度'] = confidence
        new_row['予測ID'] = prediction_id
        new_row['保存日時'] = datetime.now().isoformat()
        
        # メタデータがある場合は追加
        if metadata:
            for key, value in metadata.items():
                new_row[f'metadata_{key}'] = value
        
        # DataFrameに追加
        accumulated_df = pd.concat([accumulated_df, new_row], ignore_index=True)
        
        # CSVに保存
        accumulated_df.to_csv(self.accumulated_csv, index=False, encoding='utf-8')
        
        # メタデータの更新
        self._update_metadata(prediction_id, actual_pattern, metadata)
        
        print(f"データを蓄積しました: ID={prediction_id}, 実際のパターン={actual_pattern}")
        
    def _update_metadata(self, prediction_id: str, actual_pattern: str, metadata: Optional[Dict] = None):
        """メタデータの更新"""
        if self.metadata_json.exists():
            with open(self.metadata_json, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        all_metadata[prediction_id] = {
            'timestamp': datetime.now().isoformat(),
            'actual_pattern': actual_pattern,
            'additional_info': metadata or {}
        }
        
        with open(self.metadata_json, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    def get_accumulated_data(self) -> pd.DataFrame:
        """蓄積されたデータを取得"""
        if self.accumulated_csv.exists():
            return pd.read_csv(self.accumulated_csv, encoding='utf-8')
        else:
            return pd.DataFrame()
    
    def prepare_training_data(self, include_original: bool = True) -> pd.DataFrame:
        """学習用データの準備（元データ＋蓄積データ）"""
        training_data_list = []
        
        # 元の学習データを含める場合
        if include_original:
            original_files = list(self.training_dir.glob("*.csv"))
            for file_path in original_files:
                try:
                    # Shift-JISとUTF-8の両方を試す
                    for encoding in ['shift_jis', 'utf-8']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            training_data_list.append(df)
                            print(f"読み込み成功: {file_path.name} (encoding={encoding})")
                            break
                        except:
                            continue
                except Exception as e:
                    print(f"読み込みエラー: {file_path.name} - {e}")
        
        # 蓄積データを追加
        accumulated_df = self.get_accumulated_data()
        if not accumulated_df.empty:
            # 学習に必要なカラムのみ抽出
            required_cols = ['開始TD', '終了TD', '区間距離', '支保パターン']
            
            # 特徴量カラム
            for param in ['回転圧[MPa]', '打撃圧[MPa]', 'フィード圧[MPa]', 
                         '穿孔速度[cm/秒]', '穿孔エネルギー[J/cm^3]']:
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    required_cols.append(f"{param}_{stat}")
            
            # 必要なカラムが存在する場合のみ追加
            if all(col in accumulated_df.columns for col in required_cols):
                training_data_list.append(accumulated_df[required_cols])
                print(f"蓄積データを追加: {len(accumulated_df)}行")
        
        # すべてのデータを結合
        if training_data_list:
            combined_df = pd.concat(training_data_list, ignore_index=True)
            print(f"総学習データ数: {len(combined_df)}行")
            return combined_df
        else:
            print("学習データがありません")
            return pd.DataFrame()
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """学習データの概要を取得"""
        summary = {
            'original_files': [],
            'accumulated_count': 0,
            'total_count': 0,
            'class_distribution': {},
            'file_details': {}
        }
        
        # 元データファイル
        original_files = list(self.training_dir.glob("*.csv"))
        summary['original_files'] = [f.name for f in original_files]
        
        # 各ファイルの詳細情報を取得
        for file_path in original_files:
            try:
                # ファイル情報
                file_stat = file_path.stat()
                file_size_mb = file_stat.st_size / (1024 * 1024)
                
                # データの概要
                try:
                    # Shift-JISで試す
                    df = pd.read_csv(file_path, encoding='shift-jis', nrows=0)
                except:
                    # UTF-8で試す
                    df = pd.read_csv(file_path, encoding='utf-8', nrows=0)
                
                # 行数を効率的にカウント
                with open(file_path, 'rb') as f:
                    row_count = sum(1 for _ in f) - 1  # ヘッダー行を除く
                
                summary['file_details'][file_path.name] = {
                    'row_count': row_count,
                    'column_count': len(df.columns),
                    'file_size': f"{file_size_mb:.2f} MB",
                    'last_modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                print(f"ファイル情報取得エラー: {file_path.name} - {e}")
                summary['file_details'][file_path.name] = {
                    'row_count': '-',
                    'column_count': '-',
                    'file_size': '-',
                    'last_modified': '-'
                }
        
        # 蓄積データ数
        accumulated_df = self.get_accumulated_data()
        summary['accumulated_count'] = len(accumulated_df)
        
        # 全データの統計
        all_data = self.prepare_training_data()
        if not all_data.empty:
            summary['total_count'] = len(all_data)
            if '支保パターン' in all_data.columns:
                summary['class_distribution'] = all_data['支保パターン'].value_counts().to_dict()
        
        return summary
    
    def export_training_data(self, output_path: str, include_original: bool = True):
        """学習データをエクスポート"""
        combined_df = self.prepare_training_data(include_original)
        if not combined_df.empty:
            combined_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"学習データをエクスポートしました: {output_path}")
        else:
            print("エクスポートするデータがありません")
    
    def clear_accumulated_data(self, backup: bool = True):
        """蓄積データをクリア"""
        if backup and self.accumulated_csv.exists():
            # バックアップを作成
            backup_name = f"accumulated_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            backup_path = self.accumulated_dir / backup_name
            accumulated_df = pd.read_csv(self.accumulated_csv, encoding='utf-8')
            accumulated_df.to_csv(backup_path, index=False, encoding='utf-8')
            print(f"バックアップを作成しました: {backup_path}")
        
        # ファイルを削除
        if self.accumulated_csv.exists():
            self.accumulated_csv.unlink()
        if self.metadata_json.exists():
            self.metadata_json.unlink()
        
        print("蓄積データをクリアしました")


# 使用例
if __name__ == "__main__":
    manager = TrainingDataManager()
    
    # データの概要を表示
    summary = manager.get_training_data_summary()
    print("学習データの概要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))