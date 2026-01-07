import streamlit as st

def show_spinner(text: str):
    return st.spinner(text)

def toast_success(msg: str):
    st.toast(msg, icon="✅")

def toast_error(msg: str):
    st.toast(msg, icon="❌")

def toast_warning(msg: str):
    st.toast(msg, icon="⚠️")