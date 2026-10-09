import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ---------- Page config ----------
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

# ---------- Load models (cached across reruns) ----------
@st.cache_resource
def load_vader():
    analyzer = SentimentIntensityAnalyzer()
    analyzer.emojis.update({
        "🔥": "amazing", "💯": "perfect", "😂": "hilarious",
        "😍": "love", "💀": "dead",
        "❤️": "love", "🧡": "love", "💛": "love", "💚": "love",
        "💙": "love", "💜": "love", "🖤": "love", "🤍": "love",
    })
    analyzer.lexicon.update({
        "garbage": -2.5, "trash": -2.3, "junk": -2.0, "rubbish": -2.2,
    })
    return analyzer

@st.cache_resource
def load_roberta():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )

# ---------- Helper functions ----------
def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_vader(text, analyzer):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive', score
    elif score <= -0.05:
        return 'Negative', score
    return 'Neutral', score

def analyze_pipeline(text):
    """Two-layer pipeline: VADER first, RoBERTa only when VADER is uncertain."""
    cleaned = clean_comment(text)
    analyzer = load_vader()

    vader_label, vader_score = classify_vader(cleaned, analyzer)
    vader_confidence = abs(vader_score)

    # Layer 1 handles confident cases
    if vader_confidence >= 0.5:
        return {
            'final_label': vader_label,
            'resolved_by': 'VADER (Layer 1)',
            'vader_label': vader_label,
            'vader_score': round(vader_score, 4),
            'roberta_label': None,
            'roberta_confidence': None,
        }

    # Layer 2: RoBERTa for uncertain cases
    roberta = load_roberta()
    r = roberta(cleaned)[0]
    roberta_label = r['label'].capitalize()
    roberta_conf = round(r['score'], 3)

    if roberta_conf >= 0.7:
        final = roberta_label
        resolved_by = 'RoBERTa (Layer 2)'
    else:
        final = 'Uncertain - needs review'
        resolved_by = 'Flagged for review'

    return {
        'final_label': final,
        'resolved_by': resolved_by,
        'vader_label': vader_label,
        'vader_score': round(vader_score, 4),
        'roberta_label': roberta_label,
        'roberta_confidence': roberta_conf,
    }

# ---------- UI ----------
st.title("💬 Social Media Sentiment Analyzer")
st.markdown(
    "A two-layer sentiment pipeline: **VADER** screens quickly, "
    "**Twitter-RoBERTa** handles the ambiguous cases."
)

tab1, tab2 = st.tabs(["Single Comment", "Batch Analysis"])

# ---- Tab 1 ----
with tab1:
    st.subheader("Analyze a single comment")
    user_text = st.text_area(
        "Paste a comment here:",
        height=100,
        placeholder="e.g., This new update is incredible 🔥🔥",
    )

    if st.button("Analyze", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze_pipeline(user_text)

            st.markdown("### Result")
            st.metric("Final Label", result['final_label'])
            st.caption(f"Resolved by: **{result['resolved_by']}**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**VADER (Layer 1)**")
                st.write(f"Label: `{result['vader_label']}`")
                st.write(f"Compound: `{result['vader_score']:+.4f}`")

            with col2:
                st.markdown("**RoBERTa (Layer 2)**")
                if result['roberta_label'] is None:
                    st.write("_Not needed - VADER was confident._")
                else:
                    st.write(f"Label: `{result['roberta_label']}`")
                    st.write(f"Confidence: `{result['roberta_confidence']}`")

# ---- Tab 2 ----
with tab2:
    st.subheader("Analyze a batch of comments")
    uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df = df.dropna(subset=['text'])
            df = df[df['text'].astype(str).str.strip() != '']
            comments = df['text'].astype(str).tolist()

            st.success(f"Loaded {len(comments)} comments")

            if st.button("Run Analysis", type="primary"):
                results = []
                progress = st.progress(0)
                for i, c in enumerate(comments):
                    results.append(analyze_pipeline(c))
                    progress.progress((i + 1) / len(comments))

                df_results = pd.DataFrame(results)
                df_results.insert(0, 'comment', comments)

                # Summary
                st.markdown("### Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Final labels**")
                    st.bar_chart(df_results['final_label'].value_counts())
                with col2:
                    st.markdown("**Resolved by**")
                    st.bar_chart(df_results['resolved_by'].value_counts())

                st.markdown("### Full Results")
                st.dataframe(df_results, use_container_width=True)

                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "sentiment_results.csv",
                    "text/csv",
                )