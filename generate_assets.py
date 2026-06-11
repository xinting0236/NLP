"""
generate_assets.py  —  Run ONCE before launching app.py
Generates all interactive Plotly HTML charts + model_metrics.csv
Output folder: assets/
"""

import os, re, warnings, json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter
from io import BytesIO
import base64

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

warnings.filterwarnings("ignore")
for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Preprocessing (identical to notebook)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Load dataset
print("Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "IMDB Dataset.csv"))
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

print("Preprocessing reviews (this takes a few minutes)...")
df["tokens"]         = df["review"].apply(preprocess)
df["cleaned_review"] = df["tokens"].apply(lambda t: " ".join(t))
df["cleaned_length"] = df["cleaned_review"].apply(lambda x: len(x.split()))
print(f"Done. {len(df):,} reviews processed.")

# Train / test split
X_raw  = df["cleaned_review"]
y      = df["label"]
tokens = df["tokens"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
train_tokens = tokens[X_train_raw.index]
test_tokens  = tokens[X_test_raw.index]

# TF-IDF (matches notebook)
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf  = tfidf.transform(X_test_raw)

# Word2Vec (matches notebook)
print("Training Word2Vec...")
from gensim.models import Word2Vec
w2v = Word2Vec(sentences=list(train_tokens),
               vector_size=100, window=5, min_count=2,
               workers=4, epochs=5, seed=42)

def review_to_vec(toks, model):
    vecs = [model.wv[t] for t in toks if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

X_train_w2v = np.vstack([review_to_vec(t, w2v) for t in train_tokens])
X_test_w2v  = np.vstack([review_to_vec(t, w2v) for t in test_tokens])

# Train all 4 models
print("Training all 4 models...")
all_results = []

def evaluate_model(name, clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall":    recall_score(y_te, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_te, y_pred, zero_division=0),
        "_model":    clf,
        "_y_pred":   y_pred,
    }

nb_tfidf  = evaluate_model("Naive Bayes (TF-IDF)",  MultinomialNB(alpha=0.1),                     X_train_tfidf, X_test_tfidf, y_train, y_test)
nb_w2v    = evaluate_model("Naive Bayes (Word2Vec)", GaussianNB(),                                 X_train_w2v,   X_test_w2v,   y_train, y_test)
svm_tfidf = evaluate_model("SVM (TF-IDF)",           LinearSVC(C=1.0,max_iter=2000,random_state=42), X_train_tfidf, X_test_tfidf, y_train, y_test)
svm_w2v   = evaluate_model("SVM (Word2Vec)",         LinearSVC(C=1.0,max_iter=2000,random_state=42), X_train_w2v,   X_test_w2v,   y_train, y_test)
all_results = [nb_tfidf, nb_w2v, svm_tfidf, svm_w2v]

comparison_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in all_results])
print(comparison_df[["Model","Accuracy","Precision","Recall","F1-Score"]].to_string(index=False, float_format="{:.4f}".format))

# Save model_metrics.csv
comparison_df[["Model","Accuracy","Precision","Recall","F1-Score"]].to_csv(os.path.join(ASSETS_DIR, "model_metrics.csv"), index=False)
print("Saved: assets/model_metrics.csv")

best_result = max(all_results, key=lambda r: r["F1-Score"])

# CHART 1: Word Cloud (interactive)
print("Generating interactive word clouds...")
try:
    from wordcloud import WordCloud

    def get_top_words(df_subset, top_n=60):
        all_words = " ".join(df_subset["cleaned_review"].dropna()).split()
        top       = Counter(all_words).most_common(top_n)
        return {w: f for w, f in top}

    def make_interactive_wordcloud(freq_dict, colormap, bg_color, title):
        W, H = 1400, 700

        # Step 1: render pixel-perfect WordCloud PNG
        wc = WordCloud(
            width            = W,
            height           = H,
            background_color = bg_color,
            colormap         = colormap,
            max_words        = len(freq_dict),
            prefer_horizontal= 0.7,
            relative_scaling = 0.5,
            min_font_size    = 11,
            max_font_size    = 130,
            random_state     = 42,
            collocations     = False,
            margin           = 4,
        ).generate_from_frequencies(freq_dict)

        # Convert to base64 PNG for Plotly background
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        img_src = f"data:image/png;base64,{b64}"

        # Step 2: extract word positions for hover points
        words, freqs, xs, ys = [], [], [], []

        for (word, _), font_size, (row, col), orientation, _ in wc.layout_:
            words.append(word)
            
            # Use original frequency count
            freqs.append(freq_dict[word])

            xs.append(col)
            ys.append(H - row)

        ranks = list(range(1, len(words) + 1))

        # Step 3: build figure
        fig = go.Figure()

        # Invisible scatter — only for hover tooltips
        fig.add_trace(go.Scatter(
            x    = xs,
            y    = ys,
            mode = "markers",
            marker = dict(size=18, opacity=0),   # invisible dots
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Frequency : <b>%{customdata[1]:,}</b><br>"
                "Rank &nbsp;&nbsp;&nbsp;&nbsp;: <b>#%{customdata[2]}</b>"
                "<extra></extra>"
            ),
            customdata = list(zip(words, freqs, ranks)),
            showlegend = False,
        ))

        # WordCloud PNG as background image
        fig.add_layout_image(dict(
            source  = img_src,
            xref    = "x", yref = "y",
            x       = 0,   y    = H,
            sizex   = W,   sizey= H,
            sizing  = "stretch",
            opacity = 1,
            layer   = "below",
        ))

        fig.update_layout(
            title = dict(
                text    = title,
                x       = 0.5,
                xanchor = "center",
                font    = dict(size=22, color="#2d2d2d",
                               family="Impact, Arial Black, sans-serif"),
            ),
            xaxis = dict(showgrid=False, zeroline=False,
                         showticklabels=False, range=[0, W]),
            yaxis = dict(showgrid=False, zeroline=False,
                         showticklabels=False, range=[0, H],
                         scaleanchor="x"),
            paper_bgcolor = bg_color,
            plot_bgcolor  = bg_color,
            margin        = dict(t=65, b=10, l=10, r=10),
            height        = 600,
            hoverlabel    = dict(
                bgcolor     = "#ffffff",
                bordercolor = "#aaaaaa",
                font_size   = 13,
                font_family = "Arial",
            ),
        )
        return fig

    pos_freq = get_top_words(df[df["sentiment"]=="positive"])
    neg_freq = get_top_words(df[df["sentiment"]=="negative"])

    fig_pos = make_interactive_wordcloud(pos_freq, "winter", "#ffffff", "Interactive Word Cloud — Positive Movie Reviews")
    fig_neg = make_interactive_wordcloud(neg_freq, "YlOrRd", "#ffffff", "Interactive Word Cloud — Negative Movie Reviews")
    fig_pos.write_html(os.path.join(ASSETS_DIR, "wordcloud_positive.html"), include_plotlyjs="cdn")
    fig_neg.write_html(os.path.join(ASSETS_DIR, "wordcloud_negative.html"), include_plotlyjs="cdn")
    print("Saved: assets/wordcloud_positive.html, wordcloud_negative.html")
except ImportError:
    print("  wordcloud not installed — skipping (pip install wordcloud)")

# CHART 2: Top 20 words (uses tokens, matches notebook cells 45-46)
print("Generating top words charts...")

positive_tokens = []
for toks in df[df["sentiment"]=="positive"]["tokens"]:
    positive_tokens.extend(toks)
top_pos = pd.DataFrame(Counter(positive_tokens).most_common(20), columns=["Word","Frequency"]).sort_values("Frequency")

fig_tp = px.bar(top_pos, x="Frequency", y="Word", orientation="h",
                color="Frequency", color_continuous_scale="Viridis",
                title="Top 20 Most Frequent Words — Positive Reviews")
fig_tp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#2b1a0e", height=500)
fig_tp.write_html(os.path.join(ASSETS_DIR, "top_words_positive.html"), include_plotlyjs="cdn")

negative_tokens = []
for toks in df[df["sentiment"]=="negative"]["tokens"]:
    negative_tokens.extend(toks)
top_neg = pd.DataFrame(Counter(negative_tokens).most_common(20), columns=["Word","Frequency"]).sort_values("Frequency")

fig_tn = px.bar(top_neg, x="Frequency", y="Word", orientation="h",
                color="Frequency", color_continuous_scale="Cividis",
                title="Top 20 Most Frequent Words — Negative Reviews")
fig_tn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#2b1a0e", height=500)
fig_tn.write_html(os.path.join(ASSETS_DIR, "top_words_negative.html"), include_plotlyjs="cdn")
print(f"  Top positive word: {top_pos.iloc[-1]['Word']} ({top_pos.iloc[-1]['Frequency']:,})")
print(f"  Top negative word: {top_neg.iloc[-1]['Word']} ({top_neg.iloc[-1]['Frequency']:,})")
print("Saved: assets/top_words_positive.html, top_words_negative.html")

# CHART 3: Class distribution
print("Generating class distribution chart...")
sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]
fig_dist = px.bar(sentiment_counts, x="sentiment", y="count",
                  color="sentiment",
                  color_discrete_map={"positive":"#66c2a5","negative":"#fc8d62"},
                  text="count",
                  title="Distribution of Sentiment Classes",
                  labels={"sentiment":"Sentiment","count":"Number of Reviews"})
fig_dist.update_traces(textposition="outside")
fig_dist.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="#2b1a0e", height=450)
fig_dist.write_html(os.path.join(ASSETS_DIR, "class_distribution.html"), include_plotlyjs="cdn")
print("Saved: assets/class_distribution.html")

# CHART 4: Review length distribution (matches notebook cell 48)
print("Generating review length distribution chart...")
fig_len = go.Figure()
fig_len.add_trace(go.Histogram(
    x=df[df["sentiment"]=="positive"]["cleaned_length"],
    name="Positive", marker_color="#66c2a5", opacity=0.75, nbinsx=50
))
fig_len.add_trace(go.Histogram(
    x=df[df["sentiment"]=="negative"]["cleaned_length"],
    name="Negative", marker_color="#fc8d62", opacity=0.75, nbinsx=50
))
fig_len.update_layout(
    barmode="overlay",
    title="Review Length Distribution by Sentiment",
    xaxis_title="Number of Words (after cleaning)",
    yaxis_title="Number of Reviews",
    legend_title="Sentiment",
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#2b1a0e",
    height=480,
)
fig_len.write_html(os.path.join(ASSETS_DIR, "review_length_distribution.html"), include_plotlyjs="cdn")
print("Saved: assets/review_length_distribution.html")

# CHART 5: Confusion matrix (matches notebook cell 41)
print("Generating confusion matrix chart...")
best_predictions = best_result["_y_pred"]
cm = confusion_matrix(y_test, best_predictions)
labels = ["Negative","Positive"]
fig_cm = ff.create_annotated_heatmap(
    z=cm, x=labels, y=labels, colorscale="Blues", showscale=True
)
fig_cm.update_layout(
    title=f"Confusion Matrix — {best_result['Model']}",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
    yaxis_autorange="reversed",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#2b1a0e",
    height=480,
)
fig_cm.write_html(os.path.join(ASSETS_DIR, "confusion_matrix.html"), include_plotlyjs="cdn")
print("Saved: assets/confusion_matrix.html")

# CHART 6: Model comparison (matches notebook cell 43)
print("Generating model comparison chart...")
metrics_long = comparison_df[["Model","Accuracy","Precision","Recall","F1-Score"]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
fig_mc = px.bar(
    metrics_long, x="Model", y="Score", color="Metric",
    barmode="group",
    title="Performance Comparison of All Models",
    labels={"Model":"Model","Score":"Score"},
    range_y=[0,1],
)
fig_mc.update_layout(
    xaxis_tickangle=-15,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#2b1a0e",
    height=500,
)
fig_mc.write_html(os.path.join(ASSETS_DIR, "model_comparison.html"), include_plotlyjs="cdn")
print("Saved: assets/model_comparison.html")

best_model_name = comparison_df.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")
print("\nAll assets generated. Run: streamlit run app.py")
