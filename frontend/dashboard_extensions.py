"""
ダッシュボード拡張機能（学習・データ管理）
"""
import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List


def render_training_tab(api_url: str):
    """学習タブのレンダリング"""
    st.subheader("🤖 機械学習モデルの管理")
    
    # モデルの存在確認
    try:
        response = requests.get(f"{api_url}/ml/model/exists")
        model_exists = response.json().get('exists', False)
    except:
        model_exists = False
        st.error("APIに接続できません")
        return
    
    # データ概要の取得
    try:
        response = requests.get(f"{api_url}/ml/data/summary")
        data_summary = response.json()
    except:
        data_summary = None
    
    # データ概要の表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("モデル状態", "✅ 学習済み" if model_exists else "❌ 未学習")
        if data_summary:
            st.metric("総学習データ数", data_summary.get('total_count', 0))
    
    with col2:
        if data_summary:
            st.metric("元データファイル数", len(data_summary.get('original_files', [])))
            st.metric("蓄積データ数", data_summary.get('accumulated_count', 0))
    
    # 管理中のCSVファイル一覧
    if data_summary and data_summary.get('original_files'):
        st.write("### 管理中の学習データファイル")
        
        file_list = data_summary['original_files']
        file_details = data_summary.get('file_details', {})
        
        # ファイルごとに表示
        for filename in file_list:
            details = file_details.get(filename, {})
            
            with st.expander(f"📄 {filename}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("データ数", details.get('row_count', '-'))
                    st.metric("ファイルサイズ", details.get('file_size', '-'))
                
                with col2:
                    st.metric("カラム数", details.get('column_count', '-'))
                    st.metric("最終更新", details.get('last_modified', '-'))
                
                with col3:
                    # ファイル操作ボタン
                    if st.button(f"📥 ダウンロード", key=f"download_{filename}"):
                        try:
                            response = requests.get(
                                f"{api_url}/ml/data/download/{filename}",
                                stream=True
                            )
                            if response.status_code == 200:
                                st.download_button(
                                    label="💾 ファイルをダウンロード",
                                    data=response.content,
                                    file_name=filename,
                                    mime="text/csv",
                                    key=f"download_actual_{filename}"
                                )
                            else:
                                st.error("ダウンロードに失敗しました")
                        except Exception as e:
                            st.error(f"エラー: {str(e)}")
                    
                    if st.button(f"🗑️ 削除", key=f"delete_{filename}"):
                        if st.checkbox(f"本当に {filename} を削除しますか？", key=f"confirm_delete_{filename}"):
                            try:
                                response = requests.delete(f"{api_url}/ml/data/file/{filename}")
                                if response.status_code == 200:
                                    st.success(f"{filename} を削除しました")
                                    st.rerun()
                                else:
                                    st.error("削除に失敗しました")
                            except Exception as e:
                                st.error(f"エラー: {str(e)}")
        
        # ファイルアップロード機能
        st.divider()
        with st.expander("📤 新しい学習データファイルを追加", expanded=False):
            new_file = st.file_uploader(
                "CSVファイルを選択",
                type=['csv'],
                help="学習データディレクトリに新しいCSVファイルを追加します",
                key="add_training_file"
            )
            
            if new_file is not None:
                if st.button("アップロード", key="upload_new_file"):
                    try:
                        files = {'file': (new_file.name, new_file, 'text/csv')}
                        response = requests.post(
                            f"{api_url}/ml/data/upload",
                            files=files
                        )
                        if response.status_code == 200:
                            st.success(f"{new_file.name} をアップロードしました")
                            st.rerun()
                        else:
                            st.error(f"アップロードエラー: {response.text}")
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
    
    # クラス分布の表示
    if data_summary and data_summary.get('class_distribution'):
        st.write("### 支保パターンの分布")
        class_dist = data_summary['class_distribution']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(class_dist.keys()),
                y=list(class_dist.values()),
                text=list(class_dist.values()),
                textposition='auto',
            )
        ])
        fig.update_layout(
            xaxis_title="支保パターン",
            yaxis_title="データ数",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 学習設定
    st.write("### 学習設定")
    
    # ファイル選択機能
    st.write("#### 学習データファイルの選択")
    uploaded_training_file = st.file_uploader(
        "学習用CSVファイルを選択（オプション）",
        type=['csv'],
        help="Shift-JIS形式のCSVファイルに対応しています。選択しない場合は既存の学習データを使用します。"
    )
    
    available_features = None
    selected_features = None
    
    if uploaded_training_file is not None:
        try:
            # ファイルのプレビュー
            uploaded_training_file.seek(0)
            df_preview = pd.read_csv(uploaded_training_file, encoding='shift-jis', nrows=5)
            st.write("##### アップロードファイルのプレビュー")
            st.dataframe(df_preview, use_container_width=True)
            st.info(f"ファイル名: {uploaded_training_file.name}")
            
            # 数値型カラムを特徴量候補として抽出（支保パターン以外）
            numeric_cols = df_preview.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if '支保パターン' in numeric_cols:
                numeric_cols.remove('支保パターン')
            available_features = numeric_cols
            
            uploaded_training_file.seek(0)  # ファイルポインタをリセット
        except Exception as e:
            st.error(f"ファイルの読み込みエラー: {str(e)}")
            uploaded_training_file = None
    else:
        # デフォルトの特徴量リストを使用
        default_features = []
        for param in ['回転圧[MPa]', '打撃圧[MPa]', 'フィード圧[MPa]', 
                     '穿孔速度[cm/秒]', '穿孔エネルギー[J/cm^3]']:
            for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                default_features.append(f"{param}_{stat}")
        default_features.append('区間距離')
        available_features = default_features
    
    # 特徴量選択
    if available_features:
        st.write("#### 学習に使用する特徴量の選択")
        
        # 全選択/全解除ボタン
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("すべて選択", use_container_width=True):
                st.session_state.selected_features = available_features.copy()
        with col2:
            if st.button("すべて解除", use_container_width=True):
                st.session_state.selected_features = []
        
        # セッション状態の初期化
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = available_features.copy()
        
        # パラメータごとにグループ化して表示
        feature_groups = {}
        for feature in available_features:
            # パラメータ名を抽出
            if '_' in feature:
                param_name = feature.rsplit('_', 1)[0]
            else:
                param_name = 'その他'
            
            if param_name not in feature_groups:
                feature_groups[param_name] = []
            feature_groups[param_name].append(feature)
        
        # グループごとに表示
        for param_name, features in feature_groups.items():
            with st.expander(f"{param_name} ({len(features)}個の特徴量)", expanded=True):
                cols = st.columns(4)
                for idx, feature in enumerate(features):
                    col_idx = idx % 4
                    with cols[col_idx]:
                        is_selected = st.checkbox(
                            feature,
                            value=feature in st.session_state.selected_features,
                            key=f"feature_{feature}"
                        )
                        if is_selected and feature not in st.session_state.selected_features:
                            st.session_state.selected_features.append(feature)
                        elif not is_selected and feature in st.session_state.selected_features:
                            st.session_state.selected_features.remove(feature)
        
        selected_features = st.session_state.selected_features
        st.info(f"選択された特徴量: {len(selected_features)}個")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cv_folds = st.number_input(
            "クロスバリデーション分割数",
            min_value=2,
            max_value=10,
            value=5,
            help="データを何分割して検証するか"
        )
        test_size = st.slider(
            "テストデータの割合",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="検証用に残すデータの割合"
        )
    
    with col2:
        learning_rate = st.number_input(
            "学習率",
            min_value=0.001,
            max_value=0.3,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="学習の速度（小さいほど慎重）"
        )
        num_leaves = st.number_input(
            "葉の数",
            min_value=10,
            max_value=100,
            value=31,
            help="決定木の複雑さ（大きいほど複雑なパターンを学習可能だが過学習のリスクも増加）"
        )
    
    with col3:
        feature_fraction = st.slider(
            "特徴量サンプリング率",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="各決定木で使用する特徴量（穿孔パラメータ）の割合。ランダムに選択することで汎化性能を向上"
        )
        bagging_fraction = st.slider(
            "データサンプリング率",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="各木で使用するデータの割合"
        )
    
    include_original = st.checkbox(
        "元の学習データを含める",
        value=True,
        help="蓄積データに加えて元の学習データも使用"
    )
    
    # 学習実行ボタン
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 学習を開始", type="primary", use_container_width=True):
            # 学習状態の確認
            try:
                status_response = requests.get(f"{api_url}/ml/training/status")
                status = status_response.json()
                if status['status'] == 'training':
                    st.warning("既に学習が実行中です")
                    return
            except:
                pass
            
            # 学習の開始
            config = {
                "cv_folds": cv_folds,
                "test_size": test_size,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "include_original": include_original,
                "selected_features": selected_features if selected_features else None
            }
            
            try:
                # カスタムファイルがアップロードされている場合
                if uploaded_training_file is not None:
                    files = {
                        'training_file': (
                            uploaded_training_file.name,
                            uploaded_training_file,
                            'text/csv'
                        )
                    }
                    # selected_featuresをJSON文字列として送信
                    form_data = {k: str(v) for k, v in config.items() if k != 'selected_features'}
                    if config.get('selected_features'):
                        form_data['selected_features'] = json.dumps(config['selected_features'])
                    
                    response = requests.post(
                        f"{api_url}/ml/train/custom",
                        files=files,
                        data=form_data,
                        timeout=5
                    )
                else:
                    # 既存データで学習
                    response = requests.post(
                        f"{api_url}/ml/train",
                        json=config,
                        timeout=5
                    )
                    
                if response.status_code == 200:
                    st.success("学習を開始しました")
                else:
                    st.error(f"エラー: {response.text}")
            except Exception as e:
                st.error(f"学習の開始に失敗しました: {str(e)}")
    
    # 学習進捗の表示
    st.divider()
    st.write("### 学習進捗")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📊 学習進捗を表示", use_container_width=True):
            st.session_state.show_training_progress = True
    with col2:
        st.info("学習中の進捗状況と結果を表示します。学習が完了すると精度や特徴量重要度が確認できます。")
    
    if 'show_training_progress' not in st.session_state:
        st.session_state.show_training_progress = False
    
    if st.session_state.show_training_progress:
        display_training_progress(api_url)
    
    # データ管理
    st.divider()
    st.write("### データ管理")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 蓄積データをクリア", type="secondary"):
            if st.checkbox("本当にクリアしますか？"):
                try:
                    response = requests.post(f"{api_url}/ml/data/clear")
                    if response.status_code == 200:
                        st.success("蓄積データをクリアしました")
                        st.rerun()
                except Exception as e:
                    st.error(f"エラー: {str(e)}")


def display_training_progress(api_url: str):
    """学習進捗の表示"""
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        try:
            response = requests.get(f"{api_url}/ml/training/status")
            status = response.json()
            
            if status['status'] == 'training':
                st.write("### 学習進捗")
                progress_bar = st.progress(status['progress'] / 100)
                st.write(f"**ステップ**: {status['current_step']}")
                st.write(f"**メッセージ**: {status['message']}")
                
                # 自動更新
                time.sleep(2)
                st.rerun()
                
            elif status['status'] == 'idle' and status.get('results'):
                st.write("### 学習結果")
                results = status['results']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("テスト精度", f"{results['accuracy']:.2%}")
                with col2:
                    st.metric("CV平均スコア", f"{results['cv_score']:.2%}")
                with col3:
                    st.metric("CV標準偏差", f"±{results['cv_std']:.2%}")
                
                # 可視化画像の表示
                if 'visualization' in results:
                    st.write("### 詳細な学習結果")
                    img_data = base64.b64decode(results['visualization'])
                    st.image(img_data, use_column_width=True)
                    
        except Exception as e:
            st.error(f"進捗の取得に失敗しました: {str(e)}")


def render_prediction_with_save(api_url: str, result: Dict[str, Any], input_data: pd.DataFrame):
    """予測結果の表示と保存オプション"""
    st.subheader("🎯 予測結果")
    
    prediction = result['prediction']
    
    # 基本的な予測結果の表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("予測支保パターン", prediction['predicted_pattern'])
    
    with col2:
        confidence = prediction['confidence_score']
        st.metric("信頼度", f"{confidence:.1%}")
    
    with col3:
        st.metric("予測ID", prediction['prediction_id'])
    
    # データ蓄積オプション
    st.divider()
    st.write("### データ蓄積オプション")
    
    save_data = st.checkbox("このデータを学習用に蓄積する", value=False)
    
    if save_data:
        col1, col2 = st.columns(2)
        
        with col1:
            actual_pattern = st.selectbox(
                "実際の支保パターン",
                options=['DI-1', 'DI-6', 'DI-8', 'DIIIa-2', 'DIIIa-3'],
                help="実際に使用した支保パターンを選択"
            )
        
        with col2:
            st.info("実際の支保パターンを選択して保存すると、将来の学習データとして使用されます。")
        
        if st.button("💾 データを保存", type="primary"):
            try:
                # 入力データをJSON形式に変換
                input_json = input_data.to_json(orient='records')
                
                # 保存リクエスト
                save_data = {
                    'prediction_id': prediction['prediction_id'],
                    'actual_pattern': actual_pattern,
                    'confidence': confidence,
                    'input_data': input_json,
                    'metadata': json.dumps({
                        'tunnel_name': prediction.get('tunnel_name', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    })
                }
                
                response = requests.post(
                    f"{api_url}/ml/data/save_prediction",
                    data=save_data
                )
                
                if response.status_code == 200:
                    st.success("✅ データを保存しました")
                else:
                    st.error(f"保存エラー: {response.text}")
                    
            except Exception as e:
                st.error(f"保存に失敗しました: {str(e)}")