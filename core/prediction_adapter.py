"""
予測用のデータアダプター
既存の前処理ロジックと新しいCSV形式のブリッジ
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class PredictionAdapter:
    """異なるデータ形式を統一的に扱うアダプター"""
    
    @staticmethod
    def convert_raw_to_features(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        生の穿孔データを特徴量形式に変換
        
        Args:
            df: 生データ（測定位置、回転圧、打撃圧、etc.）
            window_size: 窓サイズ
            
        Returns:
            特徴量DataFrame（統計量形式）
        """
        features_list = []
        
        # パラメータ名のマッピング
        param_mapping = {
            '回転圧': '回転圧[MPa]',
            '打撃圧': '打撃圧[MPa]',
            'フィード圧': 'フィード圧[MPa]',
            '穿孔速度': '穿孔速度[cm/秒]',
            '穿孔エネルギー': '穿孔エネルギー[J/cm^3]'
        }
        
        # 区間ごとに統計量を計算
        for i in range(0, len(df), window_size):
            section = df.iloc[i:i+window_size]
            
            if len(section) < window_size * 0.8:
                continue
            
            section_features = {}
            
            # 開始・終了位置
            if '測定位置' in section.columns:
                section_features['開始TD'] = section['測定位置'].iloc[0]
                section_features['終了TD'] = section['測定位置'].iloc[-1]
                section_features['区間距離'] = section_features['終了TD'] - section_features['開始TD']
            
            # 各パラメータの統計量
            for raw_name, feature_name in param_mapping.items():
                if raw_name in section.columns:
                    values = section[raw_name].dropna()
                    if len(values) > 0:
                        section_features[f'{feature_name}_mean'] = values.mean()
                        section_features[f'{feature_name}_std'] = values.std()
                        section_features[f'{feature_name}_min'] = values.min()
                        section_features[f'{feature_name}_25%'] = values.quantile(0.25)
                        section_features[f'{feature_name}_50%'] = values.quantile(0.50)
                        section_features[f'{feature_name}_75%'] = values.quantile(0.75)
                        section_features[f'{feature_name}_max'] = values.max()
            
            features_list.append(section_features)
        
        return pd.DataFrame(features_list)
    
    @staticmethod
    def prepare_for_model(features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量DataFrameをモデル入力形式に整形
        
        Args:
            features_df: 特徴量DataFrame
            
        Returns:
            モデル用に整形されたDataFrame
        """
        # 必要なカラムを確認して、欠損している場合はデフォルト値で埋める
        required_features = []
        
        # 5つのパラメータ × 7統計量 = 35特徴量
        for param in ['回転圧[MPa]', '打撃圧[MPa]', 'フィード圧[MPa]', 
                     '穿孔速度[cm/秒]', '穿孔エネルギー[J/cm^3]']:
            for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                feature_name = f"{param}_{stat}"
                required_features.append(feature_name)
                
                # カラムが存在しない場合は0で埋める
                if feature_name not in features_df.columns:
                    features_df[feature_name] = 0.0
        
        # 区間距離も追加
        if '区間距離' not in features_df.columns:
            features_df['区間距離'] = 0.0
        
        return features_df
    
    @staticmethod
    def is_processed_format(df: pd.DataFrame) -> bool:
        """
        データが既に処理済み形式かどうかを判定
        
        Args:
            df: 入力DataFrame
            
        Returns:
            処理済み形式の場合True
        """
        # 統計量カラムが存在するかチェック
        stat_columns = [col for col in df.columns if any(
            stat in col for stat in ['_mean', '_std', '_min', '_25%', '_50%', '_75%', '_max']
        )]
        
        return len(stat_columns) > 20  # 統計量カラムが多数存在する場合は処理済み