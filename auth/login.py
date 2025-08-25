# auth/login.py  ✅ Self-contained 
import streamlit as st
import psycopg2, psycopg2.extras
import bcrypt, pyotp, qrcode, io

# ---------- DB Connection ----------
def get_connection():
    DB = st.secrets["auth_postgres"]
    return psycopg2.connect(
        host=DB["host"],
        dbname=DB["dbname"],
        user=DB["user"],
        password=DB["password"],
        port=DB.get("port", 5432),
        cursor_factory=psycopg2.extras.RealDictCursor
    )

def run_query(query: str, params=None, fetch: bool = False):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params or ())
    rows = cur.fetchall() if fetch else None
    conn.commit()
    cur.close()
    conn.close()
    return rows

# ---------- DB helpers ----------
def get_user(username: str):
    rows = run_query(
        "SELECT user_id, username, password_hash, totp_secret FROM auth.users WHERE username = %s;",
        (username,), fetch=True
    )
    return rows[0] if rows else None

def create_user(username: str, password_plain: str):
    exists = run_query("SELECT 1 FROM auth.users WHERE username = %s;", (username,), fetch=True)
    if exists:
        return "Username already exists."

    secret = pyotp.random_base32()
    pwd_hash = bcrypt.hashpw(password_plain.encode(), bcrypt.gensalt()).decode()
    run_query("""
        INSERT INTO auth.users(username, password_hash, totp_secret)
        VALUES (%s,%s,%s);
    """, (username, pwd_hash, secret))
    return secret

# ---------- UI components ----------
def _registration_ui():
    st.subheader("📝 Create account")
    u = st.text_input("Username", key="reg_u")
    p1 = st.text_input("Password", type="password", key="reg_p1")
    p2 = st.text_input("Confirm Password", type="password", key="reg_p2")
    if st.button("Register"):
        if not u or not p1:
            st.error("Username and password required."); return
        if p1 != p2:
            st.error("Passwords do not match."); return

        secret = create_user(u, p1)
        if isinstance(secret, str) and secret.startswith("Username"):
            st.error(secret); return

        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=u, issuer_name="Sentiment Framework")
        img = qrcode.make(uri)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        st.success("✅ Account created. Scan this QR in Google Authenticator, then go to Login:")
        st.image(buf.getvalue())

def _login_ui():
    st.subheader("🔐 Login")
    u = st.text_input("Username", key="login_u")
    p = st.text_input("Password", type="password", key="login_p")
    if st.button("Continue"):
        user = get_user(u)
        if not user:
            st.error("Invalid username."); return
        try:
            if not bcrypt.checkpw(p.encode(), user["password_hash"].encode()):
                st.error("Invalid password."); return
        except Exception:
            st.error("Password check failed."); return

        st.session_state.setdefault("auth", {})
        st.session_state["auth"]["awaiting_otp"] = True
        st.session_state["auth"]["temp_user"] = user
        st.info("Password OK ✅. Enter your 6-digit OTP below.")

    if st.session_state.get("auth", {}).get("awaiting_otp"):
        otp = st.text_input("6-digit OTP", max_chars=6, key="otp_input")
        if st.button("Verify OTP"):
            user = st.session_state["auth"]["temp_user"]
            totp = pyotp.TOTP(user["totp_secret"])
            if totp.verify(otp):
                st.session_state["auth"] = {
                    "authenticated": True,
                    "user": {"user_id": user["user_id"], "username": user["username"]},
                }
                st.success("🎉 Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid OTP.")

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state["auth"] = {"authenticated": False}
        st.rerun()

# ---------- Guard to call at top of any page ----------
def ensure_logged_in():
    st.session_state.setdefault("auth", {"authenticated": False})
    if st.session_state["auth"].get("authenticated"):
        return st.session_state["auth"]["user"]

    st.sidebar.title("User Access")
    mode = st.sidebar.radio("Select:", ["Login", "Register"])
    if mode == "Register":
        _registration_ui()
    else:
        _login_ui()
    st.stop()
