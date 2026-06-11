"""
Movie Review Sentiment Detector — Multi-page Streamlit App
"""

import os, re, warnings, sqlite3, datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")
for pkg in ["stopwords","wordnet","punkt","punkt_tab","omw-1.4"]:
    nltk.download(pkg, quiet=True)

C_DARK   = "#8A5F41"
C_BG     = "#A77F60"
C_BOX    = "#F3E4C9"
C_SIDE   = "#1E1E24"

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH = "history_reviews.db"

def init_db():
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                movie_title TEXT DEFAULT 'General',
                review TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL,
                model_used TEXT
            )
        """)
        con.commit()
        cursor = con.cursor()
        cursor.execute("PRAGMA table_info(history)")
        cols = [c[1] for c in cursor.fetchall()]
        if "movie_title" not in cols:
            con.execute("ALTER TABLE history ADD COLUMN movie_title TEXT DEFAULT 'General'")
            con.commit()
        con.close()
    except sqlite3.DatabaseError:
        # If the database file is corrupted or unreadable, remove it and rebuild
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        # Retry initialization once more with a fresh file
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                movie_title TEXT DEFAULT 'General',
                review TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL,
                model_used TEXT
            )
        """)
        con.commit()
        con.close()

def save_to_db(movie_title, review, sentiment, confidence, model_used):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO history (timestamp,movie_title,review,sentiment,confidence,model_used) VALUES (?,?,?,?,?,?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), movie_title, review, sentiment, confidence, model_used)
    )
    con.commit(); con.close()

def load_history(limit=200):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT ?", con, params=(limit,))
    con.close()
    return df

def delete_history():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM history")
    con.commit(); con.close()

init_db()

# ── OMDb poster ────────────────────────────────────────────────────────────────
OMDB_KEY = "bf1c9b03"
TITLE_MAP = {
    "流浪地球": "The Wandering Earth",
    "长城": "The Great Wall",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(title):
    search = TITLE_MAP.get(title, title)
    headers = {"User-Agent": "Mozilla/5.0"}

    def _parse(r):
        """Return (poster, year, genre, rating) from an OMDb response dict."""
        poster = r.get("Poster", "")
        return (
            poster if poster and poster != "N/A" else None,
            r.get("Year", ""),
            r.get("Genre", ""),
            r.get("imdbRating", "N/A"),
        )

    try:
        # 1️⃣  Exact title match  (t=)
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"t": search, "apikey": OMDB_KEY},
            timeout=10, headers=headers,
        ).json()
        if r.get("Response") == "True":
            poster, year, genre, rating = _parse(r)
            if poster:
                return poster, year, genre, rating
            # found the movie but no poster — try search fallback for a poster
            meta = (year, genre, rating)
        else:
            meta = None

        # 2️⃣  Search fallback  (s=) — takes first result that has a poster
        r2 = requests.get(
            "https://www.omdbapi.com/",
            params={"s": search, "apikey": OMDB_KEY, "type": "movie"},
            timeout=10, headers=headers,
        ).json()
        if r2.get("Response") == "True":
            for item in r2.get("Search", []):
                imdb_id = item.get("imdbID")
                if not imdb_id:
                    continue
                # fetch full record by imdbID for poster + metadata
                r3 = requests.get(
                    "https://www.omdbapi.com/",
                    params={"i": imdb_id, "apikey": OMDB_KEY},
                    timeout=10, headers=headers,
                ).json()
                if r3.get("Response") == "True":
                    poster, year, genre, rating = _parse(r3)
                    if poster:
                        return poster, year, genre, rating

        # 3️⃣  Return metadata without poster if we at least found the movie
        if meta:
            return None, meta[0], meta[1], meta[2]

    except Exception:
        pass

    return None, "", "", "N/A"

# ── BERT ───────────────────────────────────────────────────────────────────────
def _transformers_ok():
    """Check PyTorch + HuggingFace models are available. TF dropped in transformers 5.x."""
    try:
        import torch                                                        # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
        st.session_state.pop("_bert_err", None)
        return True
    except Exception as _e:
        st.session_state["_bert_err"] = str(_e)
        return False


# ── DistilBERT sentiment — PyTorch (transformers 5.x dropped TF support) ──────
@st.cache_resource(show_spinner=False)
def _load_bert_model():
    """Load DistilBERT tokenizer + PyTorch model via Auto classes."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mid       = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(mid)
    model     = AutoModelForSequenceClassification.from_pretrained(mid)
    model.eval()
    return tokenizer, model


def run_bert_sentiment(text: str):
    """Run DistilBERT inference with PyTorch — no pipeline."""
    import torch
    import torch.nn.functional as F
    tokenizer, model = _load_bert_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits             # shape: (1, 2)
    probs      = F.softmax(logits, dim=-1)[0]       # [neg_prob, pos_prob]
    pred_idx   = int(torch.argmax(probs))
    label      = "POSITIVE" if pred_idx == 1 else "NEGATIVE"
    confidence = float(probs[pred_idx])
    return label, confidence


# ── Translation — PyTorch MarianMT ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_translation_model(src_lang: str):
    """Load MarianMT tokenizer + PyTorch model for translation."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_map = {
        "ms": "Helsinki-NLP/opus-mt-ms-en",
        "zh": "Helsinki-NLP/opus-mt-zh-en",
    }
    mid = model_map.get(src_lang)
    if mid is None:
        raise ValueError(f"No translation model for language: {src_lang}")
    tokenizer = AutoTokenizer.from_pretrained(mid)
    model     = AutoModelForSeq2SeqLM.from_pretrained(mid)
    model.eval()
    return tokenizer, model


def translate_text(text: str, src_lang: str):
    """Translate text to English using PyTorch MarianMT — no pipeline."""
    try:
        import torch
        tokenizer, model = _load_translation_model(src_lang)
        inputs     = tokenizer([text], return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception:
        return None


# ── Sample reviews ─────────────────────────────────────────────────────────────
@st.cache_data
def load_sample_reviews():
    return [
        ("Inception",  "An absolute masterpiece! The pacing was perfect, the character development was spectacular, and the execution kept me on the edge of my seat."),
        ("Titanic",    "Don't waste your money on this. The trailer was misleading, and the actual movie was incredibly dull and uninspired."),
        ("Mat Kilau",  "Filem ini memang sangat luar biasa! Plot cerita penuh dengan kejutan dan lakonan semua watak sangat mantap dan berkualiti tinggi."),
        ("Cicakman",   "Jalan cerita sangat mengarut dan bosan. Rasa menyesal beli tiket, lakonan semua pelakon sangat kayu dan kaku."),
        ("流浪地球",   "这部电影真的太震撼了！演员演技在线，剧情环环相扣，从头到尾毫无尿点，绝对是今年最佳佳作。"),
        ("长城",       "故事讲得乱七八糟，台词生硬。特效看起来只有五毛钱水平，演技极其浮夸，令人非常失望。"),
    ]

SAMPLE_REVIEWS = load_sample_reviews()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Review Sentiment Detector", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght=700;800&family=Plus+Jakarta+Sans:wght=400;500;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{{
    background-color:{C_BOX}!important;
    color:{C_DARK};
    font-family:'Plus Jakarta Sans',sans-serif!important;
}}
[data-testid="stMainBlockContainer"]{{
    background-color:transparent!important;
    padding:40px 60px!important;
    max-width:100%!important;
}}
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="block-container"],.element-container,[data-testid="stForm"]{{
    background:transparent!important;box-shadow:none!important;border:none!important;
}}
[data-testid="stSidebar"]{{background-color:{C_SIDE}!important;border-right:none!important;}}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{{padding-top:40px!important;}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{{display:none!important;}}
[data-testid="stSidebar"] .stRadio label,[data-testid="stSidebar"] .stRadio label *{{
    color:#FFFFFF!important;font-size:0.95rem!important;
}}
[data-testid="stSidebar"] .stRadio label{{
    display:flex;align-items:center;gap:12px;
    padding:12px 24px!important;border-radius:30px!important;
    cursor:pointer;opacity:0.85;transition:all 0.2s;
    background:transparent!important;margin-bottom:8px;
}}
[data-testid="stSidebar"] .stRadio label:hover{{
    background:rgba(255,255,255,0.08)!important;opacity:1;
}}
[data-testid="stSidebar"] .stRadio [aria-checked="true"]+div label,
[data-testid="stSidebar"] .stRadio input:checked~label{{
    background:#FFFFFF!important;opacity:1!important;
    box-shadow:0 4px 12px rgba(0,0,0,0.15);
}}
[data-testid="stSidebar"] .stRadio [aria-checked="true"]+div label *,
[data-testid="stSidebar"] .stRadio input:checked~label *{{
    color:{C_SIDE}!important;font-weight:700!important;
}}
h1{{
    font-family:'Playfair Display',serif!important;
    font-size:2.4rem!important;font-weight:800!important;
    color:{C_DARK}!important;text-align:center;margin-bottom:1.8rem!important;
}}
.section-label{{
    font-size:0.9rem;font-weight:700;color:{C_DARK};
    margin-bottom:12px;margin-top:28px;
    text-transform:uppercase;letter-spacing:0.5px;
    border-bottom:1px solid rgba(138,95,65,0.15);padding-bottom:4px;
}}
.content-box,.member-card{{
    background:#FFFFFF!important;border-radius:20px!important;
    padding:30px 40px!important;margin-bottom:15px!important;
    color:{C_DARK}!important;font-size:1rem;line-height:1.6;
    box-shadow:0 4px 15px rgba(0,0,0,0.02);border:none!important;
    transition:transform 0.25s ease,box-shadow 0.25s ease!important;
}}
.content-box:hover,.member-card:hover{{
    transform:translateY(-4px);
    box-shadow:0 8px 24px rgba(138,95,65,0.08)!important;
}}
.member-card{{padding:32px 16px!important;text-align:center;}}
.member-card .name{{font-weight:700;font-size:1.05rem;margin-bottom:6px;}}
.member-card .role{{font-size:0.8rem;color:{C_BG};}}
.poster-card{{
    background:#FFFFFF!important;border-radius:16px;
    padding:14px;text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,0.06);
}}
.poster-card .poster-rating{{font-size:1.1rem;font-weight:800;color:{C_DARK};margin-top:8px;}}
.poster-card .poster-meta{{font-size:0.78rem;color:{C_BG};margin-top:4px;}}
.poster-placeholder{{
    background:{C_BOX};border-radius:12px;
    padding:40px 20px;font-size:3rem;
}}
textarea{{
    border-radius:16px!important;border:1px solid rgba(138,95,65,0.25)!important;
    background:#FFFFFF!important;color:{C_DARK}!important;padding:16px!important;
}}
[data-testid="stMetric"]{{
    background:#FFFFFF!important;border-radius:16px!important;
    padding:16px 20px!important;box-shadow:0 4px 15px rgba(0,0,0,0.02);
}}
[data-testid="stMetricLabel"]{{color:#A77F60!important;font-size:0.85rem!important;}}
[data-testid="stMetricValue"]{{color:{C_DARK}!important;font-size:1.8rem!important;font-weight:800!important;}}
[data-testid="stDataFrame"]{{border:1px solid rgba(138,95,65,0.15)!important;border-radius:12px!important;}}
.stButton>button[kind="primary"]{{
    background-color:{C_DARK}!important;color:#FFFFFF!important;
    border:none!important;border-radius:25px!important;
    font-weight:700;padding:10px 28px!important;transition:transform 0.15s ease;
}}
.stButton>button[kind="primary"]:hover{{transform:scale(1.02);background-color:#734e33!important;}}
.stButton>button[kind="secondary"]{{
    border-color:{C_DARK}!important;color:{C_DARK}!important;
    border-radius:25px!important;background:#FFFFFF!important;
}}
.stButton>button[kind="secondary"]:hover{{background:{C_BOX}!important;}}
[data-testid="stHeader"]{{background:transparent!important;}}
hr{{border:none!important;margin:12px 0!important;}}
</style>
""", unsafe_allow_html=True)

# ── Preprocessing ──────────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"[^a-z\s]","",text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = ["best_model.pkl","tfidf_vectorizer.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, f"Missing: {', '.join(missing)}"
    try:
        return joblib.load("best_model.pkl"), joblib.load("tfidf_vectorizer.pkl"), None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_dataset():
    path = "IMDB Dataset.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["label"]      = df["sentiment"].map({"positive":1,"negative":0})
    df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))
    return df

model, tfidf, load_error = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#FFFFFF;font-size:1.1rem;font-weight:800;"
        "margin-bottom:24px;padding-left:14px;'>"
        "Movie Review Sentiment Detector</h2>",
        unsafe_allow_html=True
    )
    page_sel = st.radio("nav", ["🏠 Home","🔍 Text Analyzer","📊 Data Explorer",
                                "📈 Visualizations","ℹ️ Model Info"],
                        label_visibility="collapsed")
    page = page_sel.split(" ",1)[1]

    st.markdown("<br>",unsafe_allow_html=True)
    if load_error:
        st.markdown("<span style='color:#ff6b6b;font-size:0.8rem;'>⚠ Model not loaded</span>",unsafe_allow_html=True)
        st.caption(load_error)
    else:
        st.markdown("<span style='color:#CCD67F;font-size:0.8rem;'>✓ Models loaded</span>",unsafe_allow_html=True)

    if _transformers_ok():
        st.markdown("<span style='color:#CCD67F;font-size:0.8rem;'>✓ BERT available</span>",unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#ff6b6b;font-size:0.8rem;'>✗ BERT Unavailable</span>",unsafe_allow_html=True)
        _err = st.session_state.get("_bert_err","")
        if _err:
            st.caption(_err[:120])


# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("<h1>Movie Review Sentiment Detector</h1>", unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Project Description</p>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content-box'>This application analyses movie reviews from the "
        "<b>IMDB dataset (50,000 reviews)</b> and predicts whether the sentiment is "
        "<b>Positive</b> or <b>Negative</b>. Built as part of an NLP course project "
        "exploring classical text vectorization techniques (TF-IDF, Word2Vec) and "
        "machine learning classifiers (Naive Bayes, SVM).</div>",
        unsafe_allow_html=True
    )

    st.markdown("<p class='section-label'>Problem Statement</p>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content-box'>Online reviews contain valuable opinions, but reading "
        "them manually is impractical at scale. This app automates <b>sentiment analysis</b> "
        "so users can instantly understand whether a movie review is positive or negative "
        "without reading the full text.</div>",
        unsafe_allow_html=True
    )

    st.markdown("<p class='section-label'>How to Use</p>", unsafe_allow_html=True)
    st.markdown("""
<div class='content-box'>
<b style='font-size:1.05rem;'>Step-by-Step User Manual</b>
<hr style='margin:10px 0;border-top:1px solid rgba(138,95,65,0.2);'>
<b>Step 1 — Configure Settings</b><br>
&bull; Go to <b>Text Analyzer</b> in the sidebar.<br>
&bull; Select a movie title or type a custom one.<br>
&bull; Choose your model: <b>Classical ML</b> (SVM + TF-IDF) or <b>Advanced Deep Learning</b> (DistilBERT).<br>
&bull; Language is auto-detected — Malay and Chinese are translated automatically.<br><br>
<b>Step 2 — Run Analysis</b><br>
&bull; Paste or type your review in the text area.<br>
&bull; Click <b>Test</b> on any pre-loaded sample review to auto-fill and run.<br>
&bull; Click <b>Run Analysis</b> to get results.<br><br>
<b>Step 3 — Interpret Results</b><br>
&bull; <b>Sentiment Badge</b> — 😊 POSITIVE or 😞 NEGATIVE with colour coding.<br>
&bull; <b>Confidence Score</b> — how certain the model is (e.g. 94.5%).<br>
&bull; <b>Gauge Chart</b> — animated Plotly needle showing confidence level.<br>
&bull; <b>Movie Poster</b> — fetched live from the OMDb API.<br>
&bull; <b>Word Influence Profiler</b> — shows which words drove the prediction.<br><br>
<b>Step 4 — Explore Data and Insights</b><br>
&bull; <b>Data Explorer</b> — search the IMDB dataset and track prediction history.<br>
&bull; <b>Visualizations</b> — interactive word clouds, distribution charts, confusion matrix, model comparison.<br>
&bull; <b>Model Info</b> — accuracy, F1-scores, and training details.
</div>
""", unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Group Members</p>", unsafe_allow_html=True)
    members = [
        ("Lavinia Mary",  "Data Collection & Preparation & Data Visualizations"),
        ("Bong Xin Ting", "Text Processing & NLP"),
        ("Jessie Moh",    "Web Application"),
    ]
    cols = st.columns(3)
    for col, (name, role) in zip(cols, members):
        with col:
            st.markdown(
                f"<div class='member-card'>"
                f"<div class='name'>{name}</div>"
                f"<div class='role'>{role}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════
# PAGE 2 — TEXT ANALYZER
# ════════════════════════════════════════════════════════════
elif page == "Text Analyzer":
    st.markdown("<h1>Text Analyzer</h1>", unsafe_allow_html=True)

    if "movie_dropdown_val" not in st.session_state:
        st.session_state["movie_dropdown_val"] = "Inception"
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""

    st.markdown("<p class='section-label'>Movie Title</p>", unsafe_allow_html=True)
    movie_options = ["Inception","Titanic","Mat Kilau","Cicakman","流浪地球","长城","Custom Title..."]
    sel_idx = movie_options.index(st.session_state["movie_dropdown_val"]) \
              if st.session_state["movie_dropdown_val"] in movie_options else 0

    movie_selection = st.selectbox("Movie Title", movie_options, index=sel_idx,
                                   label_visibility="collapsed")
    if movie_selection == "Custom Title...":
        movie_title = st.text_input("Enter custom title:", placeholder="e.g. Avengers: Endgame",
                                    label_visibility="collapsed")
    else:
        movie_title = movie_selection

    st.markdown("<p class='section-label'>Model Engine</p>", unsafe_allow_html=True)
    engine_opts = ["Classical ML (SVM Model)"]
    if _transformers_ok():
        engine_opts.append("Advanced Deep Learning (Transformer BERT)")
    nlp_engine = st.selectbox("Model Engine", engine_opts, label_visibility="collapsed")

    st.markdown("<p class='section-label'>Movie Review</p>", unsafe_allow_html=True)
    review_input = st.text_area(
        "review", label_visibility="collapsed",
        value=st.session_state["review_text"],
        placeholder="Enter your movie review here. Malay and Chinese are auto-translated.",
        height=140,
    )
    analyse_btn = st.button("Run Analysis", type="primary")

    if analyse_btn or st.session_state.get("auto_run_analysis", False):
        st.session_state["auto_run_analysis"] = False

        if not review_input.strip():
            st.warning("Please enter a review before running analysis.")
        elif not movie_title.strip():
            st.warning("Please specify a movie title.")
        else:
            final_text = review_input
            translated = False

            # ── Auto-detect language and translate ─────────────────────────────
            try:
                from langdetect import detect
                detected_lang = detect(review_input)[:2]
            except Exception:
                detected_lang = "en"

            if detected_lang != "en" and _transformers_ok():
                lang_names = {"ms": "Malay", "zh": "Chinese"}
                lang_label = lang_names.get(detected_lang, detected_lang.upper())
                with st.spinner(f"Detected {lang_label}. Translating..."):
                    try:
                        result = translate_text(review_input, detected_lang)
                        if result:
                            final_text = result
                            translated = True
                            st.info(f"**Translated ({lang_label} → English):** {final_text}")
                    except Exception as _te:
                        st.warning(f"Translation skipped: {_te}")

            # ── Run model ──────────────────────────────────────────────────────
            sentiment_label = None
            confidence      = None
            model_name      = None

            if nlp_engine.startswith("Advanced") and _transformers_ok():
                with st.spinner("Running DistilBERT…"):
                    try:
                        sentiment_label, confidence = run_bert_sentiment(final_text)
                        model_name = "DistilBERT (Transformers)"
                    except Exception as _be:
                        st.error(f"**DistilBERT failed** — falling back to Classical ML.  \nReason: `{_be}`")
                        nlp_engine = "Classical ML (SVM Model)"

            if sentiment_label is None:  # Classical ML path
                if load_error:
                    st.error(f"Model not loaded: {load_error}")
                    st.stop()
                tokens    = preprocess(final_text)
                cleaned   = " ".join(tokens)
                vec       = tfidf.transform([cleaned])
                pred      = model.predict(vec)[0]
                if hasattr(model, "predict_proba"):
                    confidence = float(model.predict_proba(vec)[0][pred])
                else:
                    score      = model.decision_function(vec)[0]
                    confidence = float(1 / (1 + np.exp(-abs(score))))
                sentiment_label = "POSITIVE" if pred == 1 else "NEGATIVE"
                model_name      = f"{type(model).__name__}"

            # ── Save to DB ─────────────────────────────────────────────────────
            save_to_db(movie_title, review_input[:500], sentiment_label,
                       round(confidence, 4) if confidence else None, model_name)

            st.markdown("<br>", unsafe_allow_html=True)
            poster_col, result_col = st.columns([1, 2])

            # ── Poster ─────────────────────────────────────────────────────────
            with poster_col:
                with st.spinner("Fetching poster..."):
                    poster_url, year, genre, imdb_rating = get_movie_poster(movie_title)
                if poster_url:
                    st.markdown(
                        f"<div class='poster-card'>"
                        f"<img src='{poster_url}' style='width:100%;border-radius:10px;'/>"
                        f"<div class='poster-rating'>IMDb: {imdb_rating}</div>"
                        f"<div class='poster-meta'>{year} | {genre}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='poster-card'>"
                        f"<div class='poster-placeholder'>🎬</div>"
                        f"<div class='poster-meta'>{movie_title}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # ── Results ────────────────────────────────────────────────────────
            with result_col:
                r1, r2 = st.columns(2)
                with r1:
                    if sentiment_label.upper() == "POSITIVE":
                        st.markdown(
                            "<div style='background:#D4EDDA;color:#155724;padding:14px;"
                            "border-radius:12px;text-align:center;font-weight:700;font-size:1.05rem;'>"
                            "😊 POSITIVE sentiment</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<div style='background:#F8D7DA;color:#721C24;padding:14px;"
                            "border-radius:12px;text-align:center;font-weight:700;font-size:1.05rem;'>"
                            "😞 NEGATIVE sentiment</div>", unsafe_allow_html=True)
                with r2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")

                r3, r4 = st.columns(2)
                r3.metric("Model", model_name)
                r4.metric("Movie", movie_title)

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=confidence*100,
                    title={"text":"Confidence Score","font":{"color":C_DARK,"size":14}},
                    number={"suffix":"%","font":{"color":C_DARK}},
                    gauge={
                        "axis":    {"range":[0,100],"tickcolor":C_DARK},
                        "bar":     {"color":C_DARK},
                        "bgcolor": "white",
                        "steps":   [
                            {"range":[0,50],  "color":"#ffebee"},
                            {"range":[50,75], "color":"#fff8e1"},
                            {"range":[75,100],"color":"#e8f5e9"},
                        ],
                        "threshold":{"line":{"color":"#CCD67F","width":4},"thickness":0.75,"value":80},
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color=C_DARK,
                    height=240, margin=dict(t=40,b=10,l=20,r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Word Influence Profiler — always shown, works with any engine ─────
            st.markdown("<p class='section-label'>Word Influence Profiler</p>", unsafe_allow_html=True)
            if load_error or tfidf is None or not hasattr(model, "coef_"):
                st.caption("Word influence requires the Classical ML model files (best_model.pkl + tfidf_vectorizer.pkl).")
            else:
                eng_tokens = preprocess(final_text)
                if not eng_tokens:
                    st.caption("No significant keyword tokens remaining after stopword removal.")
                else:
                    vocab = tfidf.vocabulary_
                    weights = model.coef_[0]
                    word_scores = []
                    for w in set(eng_tokens):
                        if w in vocab:
                            word_scores.append((w, weights[vocab[w]]))
                    if not word_scores:
                        st.caption("No matching tokens found in TF-IDF vocabulary.")
                    else:
                        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:6]
                        if nlp_engine.startswith("Advanced"):
                            st.caption("ℹ️ Showing classical ML word weights as reference — BERT uses contextual embeddings, not these scores.")
                        w_cols = st.columns(len(word_scores))
                        for w_col, (wd, sc) in zip(w_cols, word_scores):
                            with w_col:
                                lc = "#155724" if sc > 0 else "#721C24"
                                bc = "#D4EDDA" if sc > 0 else "#F8D7DA"
                                pl = "Positive" if sc > 0 else "Negative"
                                st.markdown(
                                    f"<div style='background:{bc};padding:14px;"
                                    f"border-radius:12px;text-align:center;'>"
                                    f"<b style='color:{lc};font-size:1rem;'>{wd}</b><br>"
                                    f"<span style='color:{lc};font-size:0.72rem;'>{pl} · {sc:.3f}</span>"
                                    f"</div>", unsafe_allow_html=True)

            with st.expander("Preprocessing Details"):
                if translated:
                    st.markdown(f"**Original:** {review_input[:120]}...")
                    st.markdown(f"**Translated:** {final_text[:120]}...")
                if "tokens" in dir() and not nlp_engine.startswith("Advanced"):
                    st.markdown(f"**Tokens after preprocessing:** {len(tokens)}")
                    preview = ", ".join(tokens[:30]) + ("..." if len(tokens) > 30 else "")
                    st.markdown(f"**Cleaned tokens:** `{preview}`")

            st.success("Review saved to history database.")

    # ── Sample reviews ─────────────────────────────────────────────────────────
    st.markdown("<br><p class='section-label'>Try a Multilingual Sample Review</p>", unsafe_allow_html=True)
    for idx, (m_title, sample_text) in enumerate(SAMPLE_REVIEWS):
        s1, s2 = st.columns([6,1])
        with s1:
            st.markdown(
                f"<div class='content-box' style='margin-bottom:8px;padding:14px 18px!important;'>"
                f"<b>[{m_title}]</b><br>{sample_text}</div>",
                unsafe_allow_html=True
            )
        with s2:
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            if st.button("Test", key=f"sample_{idx}", type="secondary"):
                st.session_state["review_text"]       = sample_text
                st.session_state["movie_dropdown_val"] = m_title
                st.session_state["auto_run_analysis"]  = True
                st.rerun()


# ════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("<h1>Data Explorer</h1>", unsafe_allow_html=True)
    tab_data, tab_history = st.tabs(["Dataset Overview","Review History"])

    with tab_data:
        dataset = load_dataset()
        if dataset is None:
            st.warning("Place `IMDB Dataset.csv` in the same folder as `app.py` and reload.")
        else:
            df = dataset
            st.markdown("<p class='section-label'>Dataset Statistics</p>", unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Reviews",    f"{len(df):,}")
            c2.metric("Positive",         f"{df['label'].sum():,}")
            c3.metric("Negative",         f"{(df['label']==0).sum():,}")
            c4.metric("Avg Words/Review", f"{df['word_count'].mean():.0f}")

            st.markdown("<p class='section-label'>Sentiment Distribution</p>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                pie = df["sentiment"].value_counts().reset_index()
                pie.columns = ["Sentiment","Count"]
                fig_pie = px.pie(pie, names="Sentiment", values="Count",
                                 color="Sentiment",
                                 color_discrete_map={"positive":C_BG,"negative":C_DARK},
                                 hole=0.4, title="Sentiment Class Proportions")
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)",font_color=C_DARK)
                st.plotly_chart(fig_pie, use_container_width=True)
            with g2:
                fig_hist = px.histogram(df, x="word_count", color="sentiment", nbins=50,
                                        color_discrete_map={"positive":C_BG,"negative":C_DARK},
                                        title="Review Word Count Distribution",
                                        labels={"word_count":"Word Count"})
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)",font_color=C_DARK)
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("<p class='section-label'>Search Reviews</p>", unsafe_allow_html=True)
            sq1,sq2 = st.columns(2)
            with sq1:
                search_q = st.text_input("Filter by keyword:", placeholder="e.g. acting, boring, director...")
            with sq2:
                sf = st.selectbox("Filter by sentiment:", ["All","positive","negative"])
            filtered = df.copy()
            if search_q:
                filtered = filtered[filtered["review"].str.contains(search_q,case=False,na=False)]
            if sf != "All":
                filtered = filtered[filtered["sentiment"]==sf]
            st.caption(f"Showing {min(20,len(filtered))} of {len(filtered):,} matching reviews")
            st.dataframe(filtered[["review","sentiment"]].head(20).reset_index(drop=True),
                         use_container_width=True)

    with tab_history:
        hist_df = load_history(200)
        if hist_df.empty:
            st.info("No prediction history yet. Analyse some reviews on the Text Analyzer page.")
        else:
            st.markdown("<p class='section-label'>Live Distribution</p>", unsafe_allow_html=True)
            hg1, hg2 = st.columns(2)
            with hg1:
                hp = hist_df["sentiment"].value_counts().reset_index()
                hp.columns = ["Sentiment","Count"]
                fig_hp = px.pie(hp, names="Sentiment", values="Count",
                                color="Sentiment",
                                color_discrete_map={"POSITIVE":C_BG,"NEGATIVE":C_DARK},
                                hole=0.3, title="Sentiment Prediction Ratio")
                fig_hp.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)",font_color=C_DARK)
                st.plotly_chart(fig_hp, use_container_width=True)
            with hg2:
                hb = hist_df["movie_title"].value_counts().reset_index()
                hb.columns = ["Movie","Count"]
                fig_hb = px.bar(hb, x="Movie", y="Count",
                                title="Tests per Movie Title",
                                color_discrete_sequence=[C_DARK])
                fig_hb.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)",font_color=C_DARK)
                st.plotly_chart(fig_hb, use_container_width=True)

            h1,h2,h3,h4 = st.columns(4)
            h1.metric("Total Analysed", len(hist_df))
            h2.metric("Positive",       (hist_df["sentiment"]=="POSITIVE").sum())
            h3.metric("Negative",       (hist_df["sentiment"]=="NEGATIVE").sum())
            avg_conf = hist_df["confidence"].dropna().mean()
            h4.metric("Avg Confidence", f"{avg_conf*100:.1f}%" if not np.isnan(avg_conf) else "N/A")

            st.markdown("<br>", unsafe_allow_html=True)
            hf1,hf2,hf3 = st.columns(3)
            with hf1:
                mv_list = ["All"] + sorted(hist_df["movie_title"].dropna().unique().tolist())
                mv_filter = st.selectbox("Filter by Movie:", mv_list)
            with hf2:
                sent_filter = st.selectbox("Filter by Sentiment:", ["All","POSITIVE","NEGATIVE"])
            with hf3:
                show_n = st.slider("Show N entries:", 5, 200, 20)

            disp = hist_df.copy()
            if mv_filter != "All":
                disp = disp[disp["movie_title"]==mv_filter]
            if sent_filter != "All":
                disp = disp[disp["sentiment"]==sent_filter]
            disp = disp.head(show_n)

            cols_show = [c for c in ["id","timestamp","movie_title","review","sentiment","confidence","model_used"] if c in disp.columns]
            ren = {"id":"#","timestamp":"Time","movie_title":"Movie","review":"Review",
                   "sentiment":"Sentiment","confidence":"Confidence","model_used":"Model"}
            st.dataframe(disp[cols_show].rename(columns=ren), use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear All History", type="secondary"):
                delete_history(); st.success("Cleared."); st.rerun()


# ════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════
elif page == "Visualizations":
    st.markdown("<h1>Visualizations & Insights</h1>", unsafe_allow_html=True)

    def render_html(path, height=550):
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                components.html(f.read(), height=height, scrolling=True)
        else:
            st.warning(f"Asset not found: `{path}` — run `python generate_assets.py` first.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Word Cloud", "Top Frequent Words",
        "Class Distribution", "Review Length Distribution",
        "Confusion Matrix", "Model Comparison"
    ])

    with tab1:
        st.markdown("<p class='section-label'>Word Cloud of Most Common Words</p>", unsafe_allow_html=True)
        st.markdown("#### Positive Reviews")
        render_html("assets/wordcloud_positive.html", 550)
        st.markdown("#### Negative Reviews")
        render_html("assets/wordcloud_negative.html", 550)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> Positive reviews cluster around words like <i>film, story, great, character, performance, love</i>,
reflecting praise for narrative and acting quality. Negative reviews concentrate on
<i>bad, plot, boring, waste, worst, terrible</i>, showing dissatisfaction with story structure and execution.
The clean separation between the two vocabularies explains why TF-IDF features perform so well —
sentiment-bearing words rarely overlap across classes, giving the classifier strong decision boundaries.
</div>
""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<p class='section-label'>Top 20 Most Frequent Words</p>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Positive Reviews")
            render_html("assets/top_words_positive.html", 520)
        with c2:
            st.markdown("#### Negative Reviews")
            render_html("assets/top_words_negative.html", 520)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> After stopword removal and lemmatization, positive reviews are dominated by high-frequency
tokens like <i>film, movie, story, character, great, time, good, make</i> (each appearing 8,000–10,000+ times),
while negative reviews show similar structural words but with more negative-polarity adjectives.
The overlap in neutral words (film, movie, time) highlights why TF-IDF's inverse-document-frequency
weighting is critical — it suppresses neutral common words and boosts discriminative ones like
<i>love, brilliant, terrible, worst</i> that genuinely separate classes.
</div>
""", unsafe_allow_html=True)

    with tab3:
        st.markdown("<p class='section-label'>Distribution of Sentiment Classes</p>", unsafe_allow_html=True)
        render_html("assets/class_distribution.html", 480)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> The IMDB dataset is perfectly balanced with exactly 25,000 positive and 25,000 negative reviews
(50% each). This balance is intentional and extremely beneficial — it means the classifier cannot achieve
high accuracy simply by predicting the majority class. Evaluation metrics like Accuracy, Precision, Recall,
and F1-Score are all meaningful and comparable. A balanced dataset also prevents gradient bias during
training, which would otherwise skew decision boundaries toward the larger class.
</div>
""", unsafe_allow_html=True)

    with tab4:
        st.markdown("<p class='section-label'>Review Length Distribution by Sentiment</p>", unsafe_allow_html=True)
        render_html("assets/review_length_distribution.html", 520)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> Both positive and negative reviews follow a similar right-skewed word-count distribution,
peaking around 150–250 words (after cleaning). There is no statistically significant length difference
between classes, which confirms that review sentiment is driven by <i>word choice</i> rather than
<i>review length</i>. The long tail of reviews exceeding 400 words represents highly detailed critiques
that tend to use more nuanced, mixed-sentiment language — these are often the most challenging cases for
the classifier and contribute disproportionately to the false-positive and false-negative rates
visible in the confusion matrix.
</div>
""", unsafe_allow_html=True)

    with tab5:
        st.markdown("<p class='section-label'>Confusion Matrix</p>", unsafe_allow_html=True)
        render_html("assets/confusion_matrix.html", 520)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> The confusion matrix shows the best model's classification performance on the 10,000-review
test set. True Positives (TP) and True Negatives (TN) dominate the diagonal, confirming strong overall
accuracy. False Negatives (FN — predicted Negative but actually Positive) typically arise from reviews
with hedged language such as "not bad" or understated praise. False Positives (FP — predicted Positive
but actually Negative) often originate from sarcastic negative reviews that use positive vocabulary
ironically. These edge cases are inherent limitations of bag-of-words approaches and motivate the use
of contextual models like DistilBERT.
</div>
""", unsafe_allow_html=True)

    with tab6:
        st.markdown("<p class='section-label'>Model Performance Comparison</p>", unsafe_allow_html=True)
        render_html("assets/model_comparison.html", 540)
        st.markdown("""
<div class='content-box'>
<b>Insight:</b> The grouped bar chart compares Accuracy, Precision, Recall, and F1-Score across all four
model configurations. <b>SVM (TF-IDF)</b> achieves the highest overall performance (~90% across all metrics),
confirming that sparse high-dimensional TF-IDF representations are well-suited for linear classifiers
on sentiment data. <b>Naive Bayes (TF-IDF)</b> performs strongly (~87%) despite its independence assumption,
benefiting from TF-IDF's ability to suppress noise. Both <b>Word2Vec</b> variants underperform relative to
TF-IDF, as averaging word vectors loses positional and co-occurrence context that matters for sentiment.
The consistent gap between TF-IDF and Word2Vec models across both classifiers validates the feature
extraction choice.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INFO
# ════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.markdown("<h1>Model Info</h1>", unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Model Explanations</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div class='content-box'><b>Naive Bayes</b><br>"
            "A probabilistic classifier based on Bayes Theorem. Assumes features (words) "
            "are independent given the class label.<br><br>"
            "- MultinomialNB (alpha=0.1) — used with TF-IDF<br>"
            "- GaussianNB — used with Word2Vec<br>"
            "- Fast to train; interpretable; works well on text</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div class='content-box'><b>Linear SVM</b><br>"
            "Support Vector Machine finds the optimal hyperplane separating "
            "positive and negative reviews in high-dimensional space.<br><br>"
            "- LinearSVC, C = 1.0, max_iter = 2000<br>"
            "- Highly effective for high-dimensional text<br>"
            "- Best overall performance in this project</div>",
            unsafe_allow_html=True
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            "<div class='content-box'><b>TF-IDF Features</b><br>"
            "Term Frequency-Inverse Document Frequency down-weights common words "
            "and highlights rare but meaningful ones.<br><br>"
            "- max_features = 20,000<br>"
            "- ngram_range = (1, 2) — unigrams + bigrams<br>"
            "- sublinear_tf = True</div>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            "<div class='content-box'><b>Word2Vec Features</b><br>"
            "Neural embedding mapping each word to a dense 100-dimensional vector. "
            "Reviews are the average of their word vectors.<br><br>"
            "- vector_size = 100, window = 5, min_count = 2<br>"
            "- Captures semantic relationships between words<br>"
            "- Trained on training set only</div>",
            unsafe_allow_html=True
        )

    _, centre_col, _ = st.columns([1, 2, 1])
    with centre_col:
        if _transformers_ok():
            try:
                get_bert_pipeline()
                bert_status = "✅ Loaded successfully"
            except Exception as _e:
                bert_status = f"⚠️ Import OK but failed: {_e}"
        else:
            bert_status = "❌ Not installed or pipeline initialization missing dependencies"
        st.markdown(
            f"<div class='content-box'><b>DistilBERT (Bonus — Advanced NLP)</b><br>"
            f"A lightweight transformer model pre-trained on SST-2 sentiment data from Hugging Face. "
            f"Uses contextual word embeddings — understands meaning based on surrounding words, "
            f"making it more robust to sarcasm and complex phrasing than bag-of-words methods.<br><br>"
            f"- Model: distilbert-base-uncased-finetuned-sst-2-english<br>"
            f"- 66M parameters, 40% smaller than BERT-base<br>"
            f"- No retraining — used for inference only<br>"
            f"- Status: <b>{bert_status}</b></div>",
            unsafe_allow_html=True
        )

    st.markdown("<p class='section-label'>Performance Metrics</p>", unsafe_allow_html=True)
    if os.path.exists("assets/model_metrics.csv"):
        mdf = pd.read_csv("assets/model_metrics.csv")
        st.dataframe(
            mdf.style.format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"}),
            use_container_width=True, hide_index=True
        )
    else:
        results_data = {
            "Model":     ["Naive Bayes (TF-IDF)","Naive Bayes (Word2Vec)","SVM (TF-IDF)","SVM (Word2Vec)"],
            "Accuracy":  ["0.8730","0.7559","0.8973","0.8547"],
            "Precision": ["0.8621","0.7517","0.8925","0.8506"],
            "Recall":    ["0.8880","0.7642","0.9034","0.8606"],
            "F1-Score":  ["0.8749","0.7579","0.8979","0.8556"],
        }
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

    st.markdown("<p class='section-label'>Training Details</p>", unsafe_allow_html=True)
    details = {
        "Dataset":            "IMDB 50,000 Movie Reviews",
        "Train / Test Split": "80% / 20% (stratified, random_state=42)",
        "Train samples":      "40,000",
        "Test samples":       "10,000",
        "Preprocessing":      "Lowercase → Remove HTML/URLs → Tokenize → Remove stopwords → Lemmatize",
        "Feature methods":    "TF-IDF (20k features, bigrams) · Word2Vec (100d, window=5)",
        "Classifiers":        "MultinomialNB (alpha=0.1) · GaussianNB · LinearSVC x2",
        "Best model saved":   "best_model.pkl + tfidf_vectorizer.pkl",
        "History database":   "history_reviews.db (SQLite, auto-created)",
    }
    st.markdown(
        "<div class='content-box'>" +
        "".join(f"<b>{k}:</b> {v}<br>" for k,v in details.items()) +
        "</div>",
        unsafe_allow_html=True
    )
