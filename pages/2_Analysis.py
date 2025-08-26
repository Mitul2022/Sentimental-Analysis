# Pages/2_Analysis.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix for torch openmp error

import streamlit as st
import sys, os, re
import pandas as pd
import nltk
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import torch
import time
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# ---- Fix Python path so we can import auth/ properly ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- Auth guard ----
from auth.login import ensure_logged_in, logout_button

# ✅ Ensure user is logged in
user = ensure_logged_in()
if user is None:
    st.stop()

# ---- Sidebar user info ----
st.sidebar.markdown(f"👋 Hello, **{user['username']}**")
logout_button()

# --- Page config
st.set_page_config(page_title="Aspect-Based Feedback Analysis", layout="wide")
st.title("🔍 Aspect-Based Review Analysis")

# --- Download NLTK Resources ---
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    return set(stopwords.words("english"))

STOP_WORDS = download_nltk_resources()

# --- Load spaCy (fast POS/noun-chunking) with safe fallback ---
@st.cache_resource
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])  # lightweight
    except Exception:
        return None

nlp_spacy = load_spacy()

# --- Load Relevance Model ---
@st.cache_resource
def load_relevance_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

relevance_model = load_relevance_model()
GENERIC_IRRELEVANT = [
    "good","bad","excellent","amazing","terrible","awful","positive","negative","neutral",
    "great","poor","awesome","nice","fantastic","perfect","horrible","decent","average"
]
irrelevant_embeddings = relevance_model.encode(GENERIC_IRRELEVANT, convert_to_tensor=True)

# 🔁 Cache for word relevance lookups
_word_sim_cache = {}
def is_relevant_word(word: str, threshold: float = 0.75) -> bool:
    if not word or len(word) < 2 or not word.isalpha():
        return False
    w = word.lower()
    if w in _word_sim_cache:
        return _word_sim_cache[w] < threshold
    word_emb = relevance_model.encode([w], convert_to_tensor=True)
    similarity = util.cos_sim(word_emb, irrelevant_embeddings).max().item()
    _word_sim_cache[w] = similarity
    return similarity < threshold

# --- Sentiment Model ---
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                    device=device, truncation=True, max_length=512)

sentiment_model = load_model()

# --- Session State ---
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None

# --- Utility ---
def map_sentiment(label):
    label = str(label).upper()
    if "LABEL_0" in label: return "Negative"
    if "LABEL_1" in label: return "Neutral"
    if "LABEL_2" in label: return "Positive"
    return "Neutral"

def sentiment_to_numerical(sentiment):
    return {"Positive": 1, "Neutral": 0, "Negative": -1}.get(sentiment, 0)

def numerical_to_sentiment(score):
    if score > 0.1: return "Positive"
    if score < -0.1: return "Negative"
    return "Neutral"

def extract_aspects_from_sentence(text, custom_aspects=None):
    if not isinstance(text, str) or not text.strip():
        return []
    if custom_aspects:
        t = text.lower()
        hits = [a.strip() for a in custom_aspects if a and a.lower() in t]
        return list(dict.fromkeys([h.title() for h in hits]))[:2]

    aspects, current = [], []
    if nlp_spacy:
        doc = nlp_spacy(text.lower())
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and is_relevant_word(token.text):
                current.append(token.text)
            else:
                if current:
                    aspects.append(" ".join(current).title()); current = []
        if current: aspects.append(" ".join(current).title())
    else:
        tokens = word_tokenize(text.lower())
        for word, tag in pos_tag(tokens):
            if tag.startswith("NN") and word.isalpha() and is_relevant_word(word):
                current.append(word)
            else:
                if current:
                    aspects.append(" ".join(current).title()); current = []
        if current: aspects.append(" ".join(current).title())

    return list(dict.fromkeys([a for a in aspects if a.lower() not in ["undefined","none","nan",""]]))[:2]

# Cache aspect extraction for repeated sentences
@lru_cache(maxsize=5000)
def cached_extract_aspects(sentence, custom_aspects=None):
    return extract_aspects_from_sentence(sentence, list(custom_aspects) if custom_aspects else None)

# --- Main Processing ---
def process_reviews(df, review_col, nps_col=None, custom_aspects=None):
    start_time = time.time()
    status_text, progress_bar, eta_text = st.empty(), st.progress(0), st.empty()
    reviews = df[review_col].astype(str).tolist()

    # Phase 1: Collect sentences
    sentences, sentence_info = [], []
    for idx, review_text in enumerate(reviews):
        try:
            sents = sent_tokenize(review_text)
        except Exception:
            sents = [review_text]
        sents = [s[:512] for s in sents if s.strip()]
        sentences.extend(sents)
        sentence_info.extend([(idx, review_text, s) for s in sents])
    total_sentences, total_reviews = len(sentences), len(reviews)
    progress_bar.progress(0.1)

    # Phase 2: Batched Sentiment
    sentiment_results = []
    BATCH_SIZE = 128 if torch.cuda.is_available() else 64
    for i in range(0, total_sentences, BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        try:
            sentiment_results.extend(sentiment_model(batch))
        except Exception:
            sentiment_results.extend([{"label": "LABEL_1", "score": 0.33} for _ in batch])
        if (i // BATCH_SIZE) % 10 == 0:
            frac = (i+len(batch))/total_sentences
            progress_bar.progress(0.1 + 0.6*frac)
            eta_left = (time.time()-start_time)/max(i+1,1) * (total_sentences-(i+1))
            eta_text.markdown(f"🧠 Sentiment {i+len(batch)}/{total_sentences} | ⏳ ~{int(eta_left)}s left")

    # Phase 3: Aspect Extraction
    full_data_rows, summary_map = [], {}
    for i,(review_idx,review_text,sentence) in enumerate(sentence_info):
        pred = sentiment_results[i]
        label, score = map_sentiment(pred["label"]), float(pred["score"])
        summary_map.setdefault(review_idx, []).append({"sentiment": label, "score": score})
        aspects = cached_extract_aspects(sentence, tuple(custom_aspects) if custom_aspects else None)
        for asp in aspects:
            full_data_rows.append({
                "Review No.": review_idx+1,
                "Review Text": review_text,
                "Aspect": asp,
                "Aspect_Sentiment": label,
                "Aspect_Sentiment_Score": score,
                "Aspect_Context": sentence
            })

    # Phase 4: Summarize
    summary_rows = []
    for review_idx, review_text in enumerate(reviews):
        sentiments = summary_map.get(review_idx, [])
        model_score = sum(sentiment_to_numerical(s["sentiment"])*s["score"] for s in sentiments)/len(sentiments) if sentiments else 0
        nps_score, nps_val = None, 0
        if nps_col and nps_col in df.columns:
            try:
                nps_score = float(df.iloc[review_idx][nps_col])
                if nps_score >= 9: nps_val = 1
                elif nps_score <= 6: nps_val = -1
            except: pass
        blended = 0.6*model_score + 0.4*nps_val
        summary_rows.append({
            "Review No.": review_idx+1,
            "Review Text": review_text,
            "NPS Score": nps_score,
            "Final_Sentiment": numerical_to_sentiment(blended)
        })

    progress_bar.empty(); eta_text.empty()
    status_text.success(f"✅ Processed {total_sentences} sentences from {total_reviews} reviews in {time.time()-start_time:.1f}s")
    return pd.DataFrame(full_data_rows), pd.DataFrame(summary_rows)

# --- Streamlit UI ---
uploaded_file = st.file_uploader("📤 Upload CSV/Excel with reviews & NPS", type=["csv","xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    review_cols = [c for c in df.columns if df[c].dtype == object]
    if not review_cols:
        st.error("No text columns found in the uploaded file.")
    else:
        review_col = st.selectbox("📝 Select review column", review_cols)

        exclude_reviews_input = st.text_area("🚫 Exclude Reviews (comma-separated, optional)")
        if exclude_reviews_input:
            exclude_list = [x.strip() for x in exclude_reviews_input.split(",") if x.strip()]
            if exclude_list:
                before_count = len(df)
                df = df[~df[review_col].astype(str).isin(exclude_list)]
                removed_count = before_count - len(df)
                st.success(f"✅ Excluded {removed_count} reviews. Remaining: {len(df)}")

        nps_col = st.selectbox("📊 Select NPS column (optional)", [None]+list(df.columns))
        category_col = st.selectbox("📂 Select category column (optional)", [None]+list(df.columns))
        if category_col and category_col in df.columns:
            cats = sorted(df[category_col].dropna().astype(str).unique())
            chosen = st.multiselect("🎯 Filter categories", cats)
            if chosen: df = df[df[category_col].astype(str).isin(chosen)]

        # --- Common Aspect Selection (instead of CSV upload) ---
        common_aspects = [
         "Quality", "Delivery", "Price", "Customer Service", 
         "Packaging", "Refund", "Order", "Website", 
         "Value", "Communication"
         ]

        custom_aspects = st.multiselect(
          "📌 Choose common aspects", 
           options=common_aspects,
           default=common_aspects  # ✅ preselect all
        )

        manual_aspects = st.text_input("✏️ Enter aspects manually (comma-separated)")
        

        if st.button("🚀 Process Data"):
            with st.spinner("Processing reviews..."):
                processed_df, summary_df = process_reviews(df, review_col, nps_col, custom_aspects)
                st.session_state.processed_data, st.session_state.summary_data = processed_df, summary_df

# --- Show Results ---
if st.session_state.summary_data is not None:
    st.subheader("📋 Review Sentiment Summary")
    st.dataframe(st.session_state.summary_data.head(10), use_container_width=True)
    st.download_button("📥 Download Full Summary Data",
                       st.session_state.summary_data.to_csv(index=False).encode("utf-8"),
                       "review_sentiment_summary.csv","text/csv")

if st.session_state.summary_data is not None and "Final_Sentiment" in st.session_state.summary_data.columns:
    dist_df = st.session_state.summary_data["Final_Sentiment"].value_counts().reset_index()
    dist_df.columns = ["Sentiment","Count"]
    if not dist_df.empty:
        color_map = {"Positive":"#4CAF50","Neutral":"#FFC107","Negative":"#F44336"}
        fig = go.Figure(go.Pie(labels=dist_df["Sentiment"],values=dist_df["Count"],hole=0.5,
                               marker=dict(colors=[color_map.get(s,"#9E9E9E") for s in dist_df["Sentiment"]],
                                           line=dict(width=0)),
                               textinfo="percent",hoverinfo="label+value+percent"))
        fig.update_layout(title="Sentiment Distribution",
                          annotations=[dict(text="Summary",x=0.5,y=0.5,showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.processed_data is not None:
    st.subheader("🔍 Aspect-Level Breakdown")
    st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
    st.download_button("📥 Download Full Aspect Data",
                       st.session_state.processed_data.to_csv(index=False).encode("utf-8"),
                       "aspect_level_breakdown.csv","text/csv")






