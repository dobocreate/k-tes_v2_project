"""
サンプルの穿孔データを生成するスクリプト
"""
import pandas as pd
import numpy as np

# サンプルデータの生成
np.random.seed(42)
n_samples = 100

data = {
    '測定位置': np.linspace(0, 10, n_samples),
    '回転圧': np.random.normal(50, 10, n_samples),
    '打撃圧': np.random.normal(100, 20, n_samples),
    'フィード圧': np.random.normal(30, 5, n_samples),
    '穿孔速度': np.random.normal(200, 30, n_samples),
    '穿孔エネルギー': np.random.normal(1000, 200, n_samples)
}

# DataFrameの作成
df = pd.DataFrame(data)

# Excelファイルとして保存
df.to_excel('sample_drilling_data.xlsx', index=False)
print("サンプルデータを 'sample_drilling_data.xlsx' として保存しました。")
print(f"データ形状: {df.shape}")
print("\nデータの先頭5行:")
print(df.head())