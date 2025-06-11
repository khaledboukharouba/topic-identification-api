import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from home_utils import logout

cookies = EncryptedCookieManager(prefix="my_app", password="secure")
if not cookies.ready():
    st.stop()

if cookies.get("logged_in") == "true" and not st.session_state.get("logged_in"):
    st.session_state.update({
        "logged_in": True,
        "email": cookies.get("email"),
        "first_name": cookies.get("first_name"),
        "last_name": cookies.get("last_name"),
        "role": cookies.get("role")
    })

if not st.session_state.get("logged_in"):
    st.warning("Not logged in. Go back to the main page.")
    #st.stop()
    st.switch_page("app.py")

st.title("🏠 Home Page")
st.write(f"Welcome **{st.session_state.first_name} {st.session_state.last_name}**")
st.write(f"Role: `{st.session_state.role}`")

# Logout button
if st.button("Logout"):
    logout(cookies)
    st.switch_page("app.py")
