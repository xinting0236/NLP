# Movie Review Sentiment Detector

A multi-page Streamlit app that classifies IMDB movie reviews as Positive or Negative using NLP and Machine Learning.

**Link**: https://nlp-project-hije2tlcgzyjyivrvmkbae.streamlit.app/

## Features
- Classical ML: SVM + TF-IDF (best model, ~90% accuracy)
- Advanced NLP: DistilBERT transformer (bonus)
- Multi-language: auto-detects and translates Malay/Chinese
- TMDb API: live movie poster fetching
- Interactive visualizations: word clouds, confusion matrix, model comparison, top frequency words, class distribution 
- SQLite history database

## Setup (Local)

```bash
pip install -r requirements.txt
python generate_assets.py   # run once — generates all charts
streamlit run app.py
```

Place these files in the same folder as `app.py`:
- `IMDB Dataset.csv`
- `best_model.pkl`
- `tfidf_vectorizer.pkl`

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (exclude large files via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Add secrets in **Settings → Secrets** if needed
5. Click **Deploy**

> **Note:** `IMDB Dataset.csv`, `best_model.pkl`, and `tfidf_vectorizer.pkl` are too large for GitHub.
> Use [Git LFS](https://git-lfs.github.com/) or host them on Google Drive and load via URL.

## Project Structure

```
├── app.py                  # Main Streamlit app
├── generate_assets.py      # One-time chart generator
├── requirements.txt
├── sample_reviews.csv
├── .streamlit/
│   └── config.toml         # Theme config
├── assets/                 # Generated HTML charts + model_metrics.csv
└── README.md
```

## Team
| Member | Role |
|--------|------|
| Lavinia Mary | Data Collection & Preparation & Visualizations |
| Bong Xin Ting | Text Processing & NLP |
| Jessie Moh | Web Application |
