# simple_dashboard.py
"""
トンネル地山評価システム - シンプルダッシュボード
StreamlitでCore APIと連携する簡易UI
"""

import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="トンネル地山評価システム",
    page_icon="🚧",
    layout="wide"
)

# セッション状態の初期化
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# API設定
API_BASE_URL = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000",
    help="Core APIのベースURL"
)

# ==================== ヘルパー関数 ====================
def check_api_health():
    """APIヘルスチェック"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_support_pattern(file_bytes, tunnel_name, prev_pattern, config):
    """支保パターン予測"""
    files = {"file": ("data.xlsx", file_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {
        "tunnel_name": tunnel_name,
        "previous_support_pattern": prev_pattern,
        "window_size": config['window_size'],
        "remove_outliers": config['remove_outliers']
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict/visualization",
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"APIエラー: {response.text}")
        return None

def display_prediction_results(result):
    """予測結果の表示"""
    prediction = result['prediction']
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "予測支保パターン",
            prediction['predicted_pattern'],
            help="最も可能性の高い支保パターン"
        )
    
    with col2:
        confidence = prediction['confidence_score']
        st.metric(
            "信頼度",
            f"{confidence:.1%}",
            delta=None if confidence > 0.7 else "低",
            delta_color="normal" if confidence > 0.7 else "inverse"
        )
    
    with col3:
        st.metric(
            "処理セクション数",
            prediction['preprocessing_stats']['sections_created']
        )
    
    with col4:
        outliers = prediction['preprocessing_stats']['outliers_removed']
        st.metric(
            "除去された外れ値",
            outliers,
            delta=f"{outliers/prediction['preprocessing_stats']['original_rows']*100:.1f}%" if outliers > 0 else None
        )
    
    # 確率分布のプロット
    st.subheader("支保パターン予測確率分布")
    prob_dist = prediction['probability_distribution']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(prob_dist.keys()),
            y=list(prob_dist.values()),
            marker_color=['red' if k == prediction['predicted_pattern'] else 'lightblue' 
                         for k in prob_dist.keys()],
            text=[f"{v:.1%}" for v in prob_dist.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        xaxis_title="支保パターン",
        yaxis_title="予測確率",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細情報の表示
    with st.expander("詳細情報を表示"):
        st.json(prediction)

def display_data_preview(df):
    """データプレビューの表示"""
    st.subheader("データプレビュー")
    
    # 基本統計情報
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**データ形状:**", df.shape)
        st.write("**カラム:**", ", ".join(df.columns.tolist()))
    
    with col2:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            st.write("**数値カラムの統計:**")
            st.dataframe(df[numeric_cols].describe().round(2))
    
    # データサンプル表示
    st.write("**データサンプル（先頭10行）:**")
    st.dataframe(df.head(10))
    
    # 穿孔エネルギーの分布（存在する場合）
    if '穿孔エネルギー' in df.columns or 'drilling_energy' in df.columns:
        energy_col = '穿孔エネルギー' if '穿孔エネルギー' in df.columns else 'drilling_energy'
        
        fig = px.histogram(
            df, 
            x=energy_col,
            nbins=50,
            title=f"{energy_col}の分布"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== メインアプリケーション ====================
def main():
    st.title("🚧 トンネル地山評価システム")
    st.markdown("穿孔データから支保パターンを予測します")
    
    # サイドバー
    with st.sidebar:
        st.header("システム設定")
        
        # APIヘルスチェック
        health = check_api_health()
        if health:
            st.success(f"✅ API接続: {health['status']}")
            if health['model_loaded']:
                st.info(f"モデルバージョン: {health['model_version']}")
            else:
                st.error("❌ モデルが読み込まれていません")
        else:
            st.error("❌ APIに接続できません")
        
        st.divider()
        
        # 前処理設定
        st.header("前処理設定")
        window_size = st.slider(
            "窓サイズ（セクション分割）",
            min_value=5,
            max_value=50,
            value=10,
            help="統計量を計算する際の窓サイズ"
        )
        
        remove_outliers = st.checkbox(
            "外れ値除去を有効化",
            value=True,
            help="穿孔エネルギーの外れ値を除去"
        )
        
        config = {
            'window_size': window_size,
            'remove_outliers': remove_outliers
        }
    
    # メインエリア
    tabs = st.tabs(["📊 予測実行", "📈 履歴表示", "ℹ️ 使い方"])
    
    # 予測実行タブ
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "穿孔データファイルを選択",
                type=['xlsx', 'xls'],
                help="Excel形式の穿孔データファイル"
            )
        
        with col2:
            tunnel_name = st.text_input(
                "トンネル名",
                value="新トンネル",
                help="予測対象のトンネル名"
            )
            
            prev_pattern = st.selectbox(
                "前回の支保パターン",
                options=['CII-b', 'CII', 'CI', 'DI', 'DIIa', 'DIIa-Au', 'DIIIa-Au'],
                index=2,  # CI
                help="直前区間の支保パターン"
            )
        
        if uploaded_file is not None:
            # データプレビュー
            df = pd.read_excel(uploaded_file)
            display_data_preview(df)
            
            st.divider()
            
            # 予測実行ボタン
            if st.button("🔍 支保パターンを予測", type="primary", use_container_width=True):
                with st.spinner("予測処理中..."):
                    # ファイルを再度読み込み（バイトとして）
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    
                    # API呼び出し
                    result = predict_support_pattern(
                        file_bytes,
                        tunnel_name,
                        prev_pattern,
                        config
                    )
                    
                    if result:
                        # 結果表示
                        display_prediction_results(result)
                        
                        # 履歴に追加
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'tunnel_name': tunnel_name,
                            'predicted_pattern': result['prediction']['predicted_pattern'],
                            'confidence': result['prediction']['confidence_score'],
                            'previous_pattern': prev_pattern
                        })
                        
                        # 成功メッセージ
                        st.success("予測が完了しました！")
    
    # 履歴表示タブ
    with tabs[1]:
        st.subheader("予測履歴")
        
        if st.session_state.prediction_history:
            # 履歴をDataFrameに変換
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # 履歴テーブル表示
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 履歴の統計
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "総予測数",
                    len(history_df)
                )
            
            with col2:
                avg_confidence = history_df['confidence'].mean()
                st.metric(
                    "平均信頼度",
                    f"{avg_confidence:.1%}"
                )
            
            with col3:
                most_common = history_df['predicted_pattern'].mode()[0]
                st.metric(
                    "最頻出パターン",
                    most_common
                )
            
            # 履歴のクリア
            if st.button("履歴をクリア", type="secondary"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("まだ予測履歴がありません")
    
    # 使い方タブ
    with tabs[2]:
        st.subheader("システムの使い方")
        
        st.markdown("""
        ### 1. データ準備
        - Excel形式（.xlsx または .xls）の穿孔データファイルを準備
        - 必要なカラム：
            - 測定位置（m）
            - 回転圧（MPa）
            - 打撃圧（MPa）
            - フィード圧（MPa）
            - 穿孔速度（mm/min）
            - 穿孔エネルギー（J）
        
        ### 2. パラメータ設定
        - **トンネル名**: 識別用の名前を入力
        - **前回の支保パターン**: 直前区間で使用された支保パターンを選択
        - **窓サイズ**: 統計量計算の区間サイズ（デフォルト: 10）
        - **外れ値除去**: 穿孔エネルギーの異常値を除去（推奨: ON）
        
        ### 3. 予測実行
        - ファイルをアップロード後、「支保パターンを予測」ボタンをクリック
        - 予測結果と信頼度が表示されます
        
        ### 4. 結果の解釈
        - **予測支保パターン**: 最も可能性の高い支保パターン
        - **信頼度**: 予測の確からしさ（70%以上が望ましい）
        - **確率分布**: 各支保パターンの予測確率
        
        ### 支保パターンの種類
        - **CII-b, CII, CI**: 比較的安定した地山
        - **DI, DIIa**: 中程度の補強が必要
        - **DIIa-Au, DIIIa-Au**: 強固な補強が必要
        """)

# ==================== アプリケーション起動 ====================
if __name__ == "__main__":
    main()

# ==================== 起動方法 ====================
"""
# 起動手順:
1. Core APIを起動
   python tunnel_core_api.py

2. Streamlitアプリを起動
   streamlit run simple_dashboard.py

3. ブラウザで http://localhost:8501 にアクセス
"""