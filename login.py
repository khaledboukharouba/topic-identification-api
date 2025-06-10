import streamlit as st
import sqlite3
import bcrypt
from streamlit_cookies_manager import EncryptedCookieManager

# ------------------ Cookie Setup ------------------ #
cookies = EncryptedCookieManager(
    prefix="my_app", password="a_very_secure_password"
)
if not cookies.ready():
    st.stop()

# ------------------ DB Setup ------------------ #
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user'))
        )
    ''')
    conn.commit()
    conn.close()

def add_user(email, password, first_name, last_name, role):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute('''
            INSERT INTO users (email, password, first_name, last_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (email, hashed_pw, first_name, last_name, role))
        conn.commit()
    except sqlite3.IntegrityError:
        st.warning("An account with this email already exists.")
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT password, role, first_name, last_name
        FROM users WHERE email = ?
    ''', (email,))
    data = c.fetchone()
    conn.close()
    if data:
        stored_pw, role, first_name, last_name = data
        if bcrypt.checkpw(password.encode(), stored_pw):
            return True, role, first_name, last_name
    return False, None, None, None

# ------------------ Streamlit UI ------------------ #
def login_page():
    st.title("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        success, role, first_name, last_name = authenticate_user(email, password)
        if success:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.session_state.first_name = first_name
            st.session_state.last_name = last_name
            st.session_state.role = role

            cookies["logged_in"] = "true"
            cookies["email"] = email
            cookies["first_name"] = first_name
            cookies["last_name"] = last_name
            cookies["role"] = role
            cookies.save()

            st.success("Login successful. Reloading...")
            st.rerun()
        else:
            st.error("Invalid email or password.")

def register_page():
    st.title("📝 Register")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    role = st.selectbox("Select Role", ["user", "admin"])

    if st.button("Create Account"):
        if first_name and last_name and email and password:
            add_user(email, password, first_name, last_name, role)
            st.success("Account created. Go to login page.")
        else:
            st.warning("Please fill in all fields.")

def home_page():
    st.title("🏠 Home Page")
    st.write(f"Welcome **{st.session_state.first_name} {st.session_state.last_name}**!")
    st.write(f"Email: `{st.session_state.email}`")
    st.write(f"Role: `{st.session_state.role}`")

    if st.session_state.role == 'admin':
        st.subheader("Admin Panel")
        st.info("This section is visible only to admin users.")

    if st.button("Logout"):
        for key in ["logged_in", "email", "role", "first_name", "last_name"]:
            cookies[key] = ""
            st.session_state.pop(key, None)
        cookies["logged_in"] = "false"
        cookies.save()
        st.rerun()

# ------------------ Main ------------------ #
def main():
    # st.set_page_config(page_title="Login App", page_icon="🔐")
    init_db()

    # Read from cookies on startup
    if 'logged_in' not in st.session_state:
        logged = cookies.get("logged_in")
        if logged == "true":
            st.session_state.logged_in = True
            st.session_state.email = cookies.get("email")
            st.session_state.first_name = cookies.get("first_name")
            st.session_state.last_name = cookies.get("last_name")
            st.session_state.role = cookies.get("role")
        else:
            st.session_state.logged_in = False

    if st.session_state.logged_in:
        home_page()
    else:
        menu = st.sidebar.radio("Menu", ["Login", "Register"])
        if menu == "Login":
            login_page()
        else:
            register_page()

if __name__ == '__main__':
    main()
