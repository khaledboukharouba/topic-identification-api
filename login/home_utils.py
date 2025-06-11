# home_utils.py
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

def logout(cookies):
    for key in ["logged_in", "email", "first_name", "last_name", "role"]:
        cookies[key] = ""
        st.session_state.pop(key, None)
    cookies["logged_in"] = "false"
    cookies.save()
    st.success("Logged out. Reloading...")
    st.rerun()
