"""
Streamlitの動作確認用テストファイル
"""
import streamlit as st

st.set_page_config(page_title="Streamlit Test", page_icon="🔧")

st.title("🔧 Streamlit動作確認")
st.write("Streamlitが正常に動作しています！")

# 簡単なインタラクティブ要素
name = st.text_input("お名前を入力してください", "ユーザー")
if st.button("挨拶"):
    st.success(f"こんにちは、{name}さん！")

st.info("このページが表示されていれば、Streamlitは正常に動作しています。")