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
import os
from typing import Dict, List, Optional, Any

# ページ設定
st.set_page_config(
    page_title="トンネル地山評価システム",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# API設定のデフォルト値
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ==================== ヘルパー関数 ====================
def check_api_health(api_url: str) -> Optional[Dict]:
    """APIヘルスチェック"""
    try:
        response = requests.get(f"{api_url}/", timeout=2)
        if response.status_code == 200:
            st.session_state.api_connected = True
            return response.json()
        st.session_state.api_connected = False
        return None
    except Exception as e:
        st.session_state.api_connected = False
        return None

def predict_support_pattern(
    api_url: str,
    file_bytes: bytes, 
    tunnel_name: str, 
    prev_pattern: str, 
    config: Dict[str, Any]
) -> Optional[Dict]:
    """支保パターン予測"""
    files = {
        "file": (
            "data.xlsx", 
            file_bytes, 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    }
    data = {
        "tunnel_name": tunnel_name,
        "previous_support_pattern": prev_pattern,
        "window_size": config['window_size'],
        "remove_outliers": config['remove_outliers']
    }
    
    try:
        response = requests.post(
            f"{api_url}/predict/visualization",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"APIエラー: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("APIリクエストがタイムアウトしました。")
        return None
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        return None

def display_prediction_results(result: Dict):
    """予測結果の表示"""
    prediction = result['prediction']
    
    # メトリクスカード
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "予測支保パターン",
            prediction['predicted_pattern'],
            help="最も可能性の高い支保パターン"
        )
    
    with col2:
        confidence = prediction['confidence_score']
        delta_color = "normal" if confidence > 0.7 else "inverse"
        st.metric(
            "信頼度",
            f"{confidence:.1%}",
            delta=None if confidence > 0.7 else "要注意",
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            "処理セクション数",
            prediction['preprocessing_stats']['sections_created'],
            help="統計量を計算したセクション数"
        )
    
    with col4:
        outliers = prediction['preprocessing_stats']['outliers_removed']
        original_rows = prediction['preprocessing_stats']['original_rows']
        outlier_rate = outliers / original_rows * 100 if original_rows > 0 else 0
        st.metric(
            "除去された外れ値",
            outliers,
            delta=f"{outlier_rate:.1f}%" if outliers > 0 else None
        )
    
    # 確率分布のプロット
    st.subheader("支保パターン予測確率分布")
    prob_dist = prediction['probability_distribution']
    
    # ソート（予測パターンを最初に）
    sorted_patterns = sorted(prob_dist.keys(), 
                           key=lambda x: (x != prediction['predicted_pattern'], x))
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_patterns,
            y=[prob_dist[k] for k in sorted_patterns],
            marker_color=['red' if k == prediction['predicted_pattern'] else 'lightblue' 
                         for k in sorted_patterns],
            text=[f"{prob_dist[k]:.1%}" for k in sorted_patterns],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        xaxis_title="支保パターン",
        yaxis_title="予測確率",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 可視化画像（APIから提供される場合）
    if 'visualization' in result and result['visualization']:
        st.subheader("詳細分析")
        img_data = base64.b64decode(result['visualization'])
        st.image(img_data, use_column_width=True)
    
    # 詳細情報の表示
    with st.expander("詳細情報を表示"):
        # 前処理統計
        if 'missing_values' in prediction['preprocessing_stats']:
            st.write("**欠損値情報:**")
            st.json(prediction['preprocessing_stats']['missing_values'])
        
        # 特徴量重要度
        if prediction.get('feature_importance'):
            st.write("**特徴量重要度 (上位10):**")
            importance_df = pd.DataFrame(
                list(prediction['feature_importance'].items()),
                columns=['特徴量', '重要度']
            ).sort_values('重要度', ascending=False).head(10)
            st.dataframe(importance_df, hide_index=True)
        
        # 完全な予測結果
        st.write("**完全な予測結果:**")
        st.json(prediction)

def display_data_preview(df: pd.DataFrame):
    """データプレビューの表示"""
    st.subheader("📊 データプレビュー")
    
    # 基本統計情報
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**データ形状:**", f"{df.shape[0]:,} 行 × {df.shape[1]} 列")
        st.write("**カラム:**", ", ".join(df.columns.tolist()))
        
        # 欠損値情報
        missing_info = df.isnull().sum()
        if missing_info.any():
            st.write("**欠損値のあるカラム:**")
            missing_df = pd.DataFrame({
                'カラム': missing_info[missing_info > 0].index,
                '欠損数': missing_info[missing_info > 0].values
            })
            st.dataframe(missing_df, hide_index=True)
    
    with col2:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            st.write("**数値カラムの統計:**")
            st.dataframe(
                df[numeric_cols].describe().round(2),
                use_container_width=True
            )
    
    # データサンプル表示
    st.write("**データサンプル（先頭10行）:**")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # 主要な特徴量の分布
    energy_cols = ['穿孔エネルギー', '削孔エネルギー', 'drilling_energy', 'Drilling Energy']
    energy_col = next((col for col in energy_cols if col in df.columns), None)
    
    if energy_col:
        fig = px.histogram(
            df[df[energy_col].notna()], 
            x=energy_col,
            nbins=50,
            title=f"{energy_col}の分布",
            labels={energy_col: f"{energy_col} (J)"}
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            height=350
        )
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        st.plotly_chart(fig, use_container_width=True)

# ==================== メインアプリケーション ====================
def main():
    st.title("🚧 トンネル地山評価システム")
    st.markdown("穿孔データから機械学習により支保パターンを予測します")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ システム設定")
        
        # API URL設定
        api_url = st.text_input(
            "API URL", 
            value=DEFAULT_API_URL,
            help="Core APIのベースURL"
        )
        
        # APIヘルスチェック
        if st.button("🔄 API接続確認", use_container_width=True):
            with st.spinner("接続確認中..."):
                health = check_api_health(api_url)
                st.session_state.last_health = health
        else:
            health = st.session_state.get('last_health', None)
            
        if health:
            st.success(f"✅ API接続: {health['status'].upper()}")
            if health['model_loaded']:
                st.info(f"📊 モデル: v{health['model_version']}")
                if health['last_updated']:
                    st.caption(f"更新: {health['last_updated']}")
            else:
                st.error("❌ モデルが読み込まれていません")
                st.info("モデルファイルを配置してAPIを再起動してください")
        elif health is False:  # 明示的に接続失敗の場合のみ
            st.error("❌ APIに接続できません")
            st.info("APIが起動していることを確認してください")
        else:
            st.info("🔄 API接続確認ボタンをクリックして接続状態を確認してください")
        
        st.divider()
        
        # 前処理設定
        st.header("🔧 前処理設定")
        window_size = st.slider(
            "窓サイズ（セクション分割）",
            min_value=5,
            max_value=50,
            value=10,
            help="統計量を計算する際の窓サイズ（データ点数）"
        )
        
        remove_outliers = st.checkbox(
            "外れ値除去を有効化",
            value=True,
            help="穿孔エネルギーの異常値を除去（IQR法）"
        )
        
        config = {
            'window_size': window_size,
            'remove_outliers': remove_outliers
        }
        
        # 支保パターン一覧
        st.divider()
        st.header("📋 支保パターン説明")
        with st.expander("支保パターンの詳細"):
            st.markdown("""
            - **CII-b**: 最も軽い支保（安定地山）
            - **CII**: 軽い支保
            - **CI**: 標準的な支保
            - **DI**: やや重い支保
            - **DIIa**: 重い支保
            - **DIIa-Au**: 重い支保＋補助工法
            - **DIIIa-Au**: 最も重い支保＋補助工法
            """)
    
    # 拡張機能のインポート
    try:
        from dashboard_extensions import render_training_tab, render_prediction_with_save
        extensions_available = True
    except ImportError:
        extensions_available = False
    
    # メインエリア
    if extensions_available:
        tabs = st.tabs(["📊 予測実行", "🤖 学習管理", "📈 履歴表示", "ℹ️ 使い方"])
    else:
        tabs = st.tabs(["📊 予測実行", "📈 履歴表示", "ℹ️ 使い方"])
    
    # 予測実行タブ
    with tabs[0]:
        if not st.session_state.get('api_connected', False):
            st.info("ℹ️ APIに接続されていません。サイドバーで「API接続確認」ボタンをクリックしてください。")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "穿孔データファイルを選択",
                type=['xlsx', 'xls'],
                help="Excel形式の穿孔データファイル（必須カラム: 測定位置, 回転圧, 打撃圧, フィード圧, 穿孔速度, 穿孔エネルギー）"
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
            try:
                # データプレビュー
                df = pd.read_excel(uploaded_file)
                display_data_preview(df)
                
                st.divider()
                
                # 予測実行ボタン
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button(
                        "🔍 支保パターンを予測", 
                        type="primary", 
                        use_container_width=True,
                        disabled=not st.session_state.api_connected
                    ):
                        with st.spinner("予測処理中..."):
                            # ファイルを再度読み込み（バイトとして）
                            uploaded_file.seek(0)
                            file_bytes = uploaded_file.read()
                            
                            # API呼び出し
                            result = predict_support_pattern(
                                api_url,
                                file_bytes,
                                tunnel_name,
                                prev_pattern,
                                config
                            )
                            
                            if result:
                                # 結果表示
                                st.divider()
                                st.subheader("🎯 予測結果")
                                display_prediction_results(result)
                                
                                # 履歴に追加
                                st.session_state.prediction_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'tunnel_name': tunnel_name,
                                    'predicted_pattern': result['prediction']['predicted_pattern'],
                                    'confidence': result['prediction']['confidence_score'],
                                    'previous_pattern': prev_pattern,
                                    'sections': result['prediction']['preprocessing_stats']['sections_created']
                                })
                                
                                # 成功メッセージ
                                st.success("✅ 予測が完了しました！")
                                
                                # ダウンロードボタン
                                result_json = json.dumps(result['prediction'], ensure_ascii=False, indent=2)
                                st.download_button(
                                    label="📥 結果をダウンロード (JSON)",
                                    data=result_json,
                                    file_name=f"prediction_{tunnel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
            
            except Exception as e:
                st.error(f"ファイルの読み込みエラー: {str(e)}")
                st.info("Excelファイルの形式を確認してください。")
    
    # 学習管理タブ（拡張機能が利用可能な場合）
    if extensions_available:
        with tabs[1]:
            render_training_tab(api_url)
    
    # 履歴表示タブ
    history_tab_index = 2 if extensions_available else 1
    with tabs[history_tab_index]:
        st.subheader("📜 予測履歴")
        
        if st.session_state.prediction_history:
            # 履歴をDataFrameに変換
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # サマリー統計
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col4:
                high_confidence_rate = (history_df['confidence'] > 0.7).mean()
                st.metric(
                    "高信頼度率",
                    f"{high_confidence_rate:.1%}",
                    help="信頼度70%以上の割合"
                )
            
            st.divider()
            
            # 履歴テーブル表示
            st.dataframe(
                history_df.sort_values('timestamp', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("日時"),
                    "tunnel_name": st.column_config.TextColumn("トンネル名"),
                    "predicted_pattern": st.column_config.TextColumn("予測パターン"),
                    "confidence": st.column_config.ProgressColumn(
                        "信頼度",
                        min_value=0,
                        max_value=1,
                        format="%.1%"
                    ),
                    "previous_pattern": st.column_config.TextColumn("前回パターン"),
                    "sections": st.column_config.NumberColumn("セクション数")
                }
            )
            
            # 履歴の管理
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ 履歴をクリア", type="secondary"):
                    st.session_state.prediction_history = []
                    st.rerun()
            
            with col2:
                # CSVダウンロード
                csv = history_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 履歴をダウンロード (CSV)",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("まだ予測履歴がありません。予測実行タブから開始してください。")
    
    # 使い方タブ
    usage_tab_index = 3 if extensions_available else 2
    with tabs[usage_tab_index]:
        st.subheader("📖 システムの使い方")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 1. データ準備
            Excel形式（.xlsx または .xls）の穿孔データファイルを準備します。
            
            **必要なカラム:**
            - 測定位置（m）
            - 回転圧（MPa）
            - 打撃圧（MPa）
            - フィード圧（MPa）
            - 穿孔速度（mm/min）
            - 穿孔エネルギー（J）
            
            ※カラム名のバリエーション（例: "削孔エネルギー"）にも対応
            
            ### 2. パラメータ設定
            - **トンネル名**: 識別用の名前を入力
            - **前回の支保パターン**: 直前区間で使用された支保パターンを選択
            - **窓サイズ**: 統計量計算の区間サイズ（デフォルト: 10）
            - **外れ値除去**: 穿孔エネルギーの異常値を除去（推奨: ON）
            """)
        
        with col2:
            st.markdown("""
            ### 3. 予測実行
            1. ファイルをアップロード
            2. データプレビューで内容を確認
            3. 「支保パターンを予測」ボタンをクリック
            4. 予測結果と信頼度が表示されます
            
            ### 4. 結果の解釈
            - **予測支保パターン**: 最も可能性の高い支保パターン
            - **信頼度**: 予測の確からしさ（70%以上が望ましい）
            - **確率分布**: 各支保パターンの予測確率
            - **処理セクション数**: データから作成された区間数
            
            ### 5. 注意事項
            - モデルファイルが必要です（APIサーバー側）
            - 大きなファイルは処理時間がかかる場合があります
            - 外れ値が多い場合は手動で確認することを推奨
            """)
        
        st.divider()
        
        # システム情報
        st.subheader("🔧 システム情報")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown("""
            **バージョン情報:**
            - Frontend: v1.0.0
            - Streamlit: v1.29.0
            - Plotly: v5.18.0
            """)
        
        with info_col2:
            st.markdown("""
            **推奨環境:**
            - Python 3.9以上
            - Chrome/Edge/Safari最新版
            - 画面解像度: 1280×720以上
            """)

# ==================== アプリケーション起動 ====================
if __name__ == "__main__":
    main()