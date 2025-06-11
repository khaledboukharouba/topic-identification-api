import streamlit as st
import sqlite3
import bcrypt
from streamlit_cookies_manager import EncryptedCookieManager
from home_utils import logout


cookies = EncryptedCookieManager(prefix="my_app", password="secure")
if not cookies.ready():
    st.stop()

def authenticate_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password, role, first_name, last_name FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True, row[1], row[2], row[3]
    return False, None, None, None

def login():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        ok, role, fn, ln = authenticate_user(email, password)
        if ok:
            st.session_state.update({
                "logged_in": True,
                "email": email,
                "first_name": fn,
                "last_name": ln,
                "role": role
            })
            cookies.update({
                "logged_in": "true",
                "email": email,
                "first_name": fn,
                "last_name": ln,
                "role": role
            })
            cookies.save()
            st.success("Login successful. Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials.")

def main():
    #st.set_page_config(page_title="Login App")
    
    if cookies.get("logged_in") == "true" and not st.session_state.get("logged_in"):
        st.session_state.update({
            "logged_in": True,
            "email": cookies.get("email"),
            "first_name": cookies.get("first_name"),
            "last_name": cookies.get("last_name"),
            "role": cookies.get("role")
        })

    if st.session_state.get("logged_in"):
        st.success("You're already logged in.")
        if st.button("Logout"):
            logout(cookies)
        else:
            st.switch_page("pages/home.py")
    else:
        login()

if __name__ == "__main__":
    main()
