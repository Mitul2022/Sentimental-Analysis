# app.py
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from auth.login import ensure_logged_in, logout_button

st.set_page_config(page_title="Review Intelligence Hub", layout="wide")

# ---- guard ----
user = ensure_logged_in()  # Will stop execution if not logged in

if user:  # only if authenticated
    st.sidebar.markdown(f"👋 Hello, **{user['username']}**")
    logout_button()

    # ---- your existing home UI ----
    def load_lottie_url(url):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None

    animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_qp1q7mct.json")

    st.sidebar.title("🧭 Navigation")
    st.sidebar.page_link("Pages/2_Analysis.py", label="⚙️ Analyze Data")
    st.sidebar.page_link("Pages/3_Report.py", label="📊 Report")

    st.markdown("<h1 style='color:#1f77b4;'>🧠 Review Intelligence Hub</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        if animation:
            st_lottie(animation, speed=1, height=250, key="welcome")
        else:
            st.warning("⚠️ Could not load animation.")
    with col2:
        st.markdown("""
        <div style='font-size:16px;'>
            Welcome to <strong>Sentiment.app</strong>, your NLP-powered solution for analyzing textual reviews.<br><br>
            <strong>🔍 Features:</strong><br>
            • Aspect-level sentiment<br>
            • Word frequency & topics<br>
            • Visualizations & reports<br><br>
            <span style='color:#888;'>Use the sidebar to move between pages</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🔧 How This App Works"):
        st.markdown("""
        ### Workflow Steps:
        1. 📂 **Upload your dataset** (CSV/Excel with reviews & optional NPS) on the **Analyze Data** page.  
        2. 📝 **Select the review column** (and optionally NPS, category, and filter categories).  
        3. ⚙️ **Click Process Data** to run the built-in NLP sentiment analysis model.  
        4. 🧩 **Extract aspect sentiments** (either upload aspects file or enter them manually).  
        5. 📊 **Visualize insights** with charts, word clouds, KPIs, and sentiment breakdowns.  
        6. 📥 **Generate & download reports** (Excel/PDF) from the **Report** page.  
        """)

    st.markdown("---")
    st.markdown("### 👉 Ready to Begin?")
    if st.button("🚀 Proceed to Analyze", use_container_width=True):
        st.switch_page("Pages/2_Analysis.py")
