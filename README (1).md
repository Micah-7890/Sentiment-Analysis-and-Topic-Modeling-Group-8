# 📰 Media Sentiment Intelligence Dashboard
### US/Israel-Iran War — Media Framing & Public Comment Analysis
**Group 8 | NLP Sentiment Analysis Project**

---

## 🚀 How to Deploy on Streamlit

### Option 1: Run Locally (Recommended for Presentation)

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Run the app**
```bash
streamlit run app.py
```

**Step 3 — Upload your data files in the sidebar**
- `master_updated1.csv` → articles
- `Master comments data1.csv` → comments

---

### Option 2: Deploy on Streamlit Cloud (Free, Shareable Link)

1. Push your project folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ For Streamlit Cloud: your CSV files must also be in the repo (rename them to avoid spaces):
> - `master_updated1.csv`
> - `Master_comments_data1.csv`

---

## 📁 Project Structure

```
sentiment_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🔬 What This App Does

| Module | Description |
|---|---|
| **Sentiment Overview** | VADER scores per outlet — articles & comments |
| **5 War Angles** | Military, Geopolitics, Economy, Media, Support — keyword-based sentiment |
| **Topic Modelling** | LDA with 7 latent topics + word clouds |
| **Articles vs Comments** | Side-by-side comparison of journalism vs public reaction |
| **Entity Tone** | Iran, Israel, US, Trump — who gets talked about most negatively? |
| **Media Framing** | Conflict / Humanitarian / Diplomacy / Legal frames |
| **Raw Data** | Explore and download all processed datasets |

---

## ⚙️ Technical Stack

- **Streamlit** — Web application framework
- **VADER** — Sentiment analysis (rule-based, optimised for social media)
- **LDA (scikit-learn)** — Latent Dirichlet Allocation topic modelling
- **Plotly** — Interactive charts
- **NLTK** — Tokenisation, lemmatisation, POS tagging
- **WordCloud** — Topic visualisation

---

## 📊 Data Requirements

| File | Required Columns |
|---|---|
| Articles CSV | `SOURCE`, `TEXT` |
| Comments CSV | `source`, `comment_text` |
