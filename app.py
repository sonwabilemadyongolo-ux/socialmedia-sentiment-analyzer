import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ---------- Page config ----------
st.set_page_config(
    page_title="SentimentAnalyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .stApp {background-color: #FDF6EC;}

    /* Nav bar */
    .nav-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 4px 0 20px 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 28px;
    }
    .nav-left {display: flex; align-items: center; gap: 12px;}
    .nav-logo {
        width: 64px; height: 64px; background: #CCFBF1;
        border-radius: 18px; display: flex; align-items: center;
        justify-content: center; font-size: 38px;
    }
    }
    .nav-title {font-size: 22px; font-weight: 800; color: #1F2937; margin: 0;}
    .nav-subtitle {font-size: 12px; color: #6B7280; margin: 0;}
    .nav-links {display: flex; gap: 26px; font-size: 14px; color: #4B5563; align-items: center;}
    .nav-link-active {color: #14B8A6; font-weight: 600;}

    /* Hero */
    .hero-title {
        font-size: 40px; font-weight: 700; line-height: 1.15;
        color: #111827; margin: 0 0 14px 0;
    }
    .hero-title .accent {
background: linear-gradient(135deg, #14B8A6, #0EA5E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
    .hero-subtitle {
        font-size: 15px; color: #4B5563; line-height: 1.6;
        margin: 0 0 26px 0; max-width: 600px;
    }

    /* Card titles */
    .card-title {
        font-size: 14px; font-weight: 600; color: #1F2937;
        margin: 0 0 4px 0;
    }
    .card-subtitle {
        font-size: 12px; color: #6B7280;
        margin: 0 0 12px 0;
    }

    /* Section heading */
    .section-title {
        font-size: 15px; font-weight: 600; color: #111827;
        margin: 0 0 14px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 10px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .metric-emoji {
font-size: 26px; line-height: 1;
    width: 52px; height: 52px;
    margin: 0 auto 10px auto;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: #F0FDFA;
}
    .metric-label {font-size: 13px; color: #374151; font-weight: 500; margin-bottom: 6px;}
    .metric-value {font-size: 26px; font-weight: 700; color: #111827; line-height: 1;}
    .metric-pct {font-size: 12px; color: #6B7280; margin-top: 4px;}

    /* Info box */
    .info-box {
        background: #F0FDFA;
        border-radius: 10px;
        padding: 14px 16px;
        color: #0F766E;
        font-size: 13px;
        line-height: 1.55;
        display: flex; gap: 10px; align-items: flex-start;
        margin: 18px 0 26px 0;
    }

    /* Feature cards */
    .feature-card {text-align: center; padding: 8px 4px;}
    .feature-icon {font-size: 22px; margin-bottom: 8px;}
    .feature-title {font-size: 12px; color: #94A3B8; line-height: 1.45;}

    /* Streamlit components */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #E5E7EB !important;
        font-size: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #14B8A6 !important;
        box-shadow: 0 0 0 3px rgba(20,184,166,0.15) !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        border: 1px solid #E5E7EB !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #14B8A6 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0D9488 !important;
    }

    /* Results table */
    .result-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px;
        border-bottom: 1px solid #F3F4F6;
        font-size: 14px;
    }
    .result-badge {
        font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 999px;
        white-space: nowrap;
    }
    .badge-positive {background: #DCFCE7; color: #166534;}
    .badge-neutral  {background: #FEF3C7; color: #92400E;}
    .badge-negative {background: #FEE2E2; color: #991B1B;}
    .badge-review   {background: #E5E7EB; color: #374151;}
    .result-comment {color: #374151; flex: 1;}

    /* Rich analysis card */
    .analysis-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .analysis-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        color: #6B7280;
        text-transform: uppercase;
        margin: 0 0 16px 0;
    }

    /* Donut */
    .donut-wrap {display: flex; align-items: center; gap: 22px;}
    .donut {
        width: 130px; height: 130px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        position: relative;
        flex-shrink: 0;
    }
    .donut-inner {
        width: 96px; height: 96px;
        background: #FFFFFF;
        border-radius: 50%;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
    }
    .donut-value {font-size: 24px; font-weight: 700; color: #111827; line-height: 1;}
    .donut-label {
        font-size: 9px; color: #6B7280;
        letter-spacing: 0.6px; margin-top: 4px;
        text-transform: uppercase;
    }
    .donut-sentiment {font-size: 32px; font-weight: 700; line-height: 1.1; margin: 0;}
    .donut-sub {font-size: 13px; color: #6B7280; margin: 4px 0 10px 0;}

    /* Gradient bar */
    .grad-bar {
        height: 8px; border-radius: 4px; position: relative;
        background: linear-gradient(to right, #ef4444, #9ca3af, #10b981);
    }
    .grad-marker {
        position: absolute; top: -4px;
        width: 3px; height: 16px; background: #111827;
        border-radius: 2px;
        transform: translateX(-50%);
    }
    .grad-labels {
        display: flex; justify-content: space-between;
        font-size: 11px; color: #6B7280; margin-top: 6px;
    }

    /* Emotion bars */
    .emo-row {margin-bottom: 14px;}
    .emo-head {
        display: flex; justify-content: space-between;
        font-size: 13px; color: #374151; margin-bottom: 6px;
    }
    .emo-bar {
        height: 6px; border-radius: 3px;
        background: #F3F4F6; overflow: hidden;
    }
    .emo-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(to right, #14B8A6, #0EA5E9);
    }

    /* Intensity meter */
    .intensity-value {
        font-size: 30px; font-weight: 700; color: #111827;
        line-height: 1; margin: 0;
    }
    .intensity-label {font-size: 14px; color: #6B7280; margin-left: 8px;}
    .intensity-bar {
        display: flex; gap: 6px; margin-top: 16px;
    }
    .intensity-seg {
        flex: 1; height: 6px; border-radius: 3px;
        background: #E5E7EB;
    }
    .intensity-seg.active {background: linear-gradient(to right, #14B8A6, #0EA5E9);}
    .intensity-labels {
        display: flex; justify-content: space-between;
        font-size: 11px; color: #6B7280; margin-top: 6px;
    }

    /* Tone pills */
    .pill-row {display: flex; flex-wrap: wrap; gap: 8px;}
    .pill {
        padding: 6px 14px;
        border-radius: 999px;
        background: #F0FDFA;
        color: #0F766E;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid #DDD6FE;
    }
    .pill-muted {background: #F3F4F6; color: #6B7280; border-color: #E5E7EB;}

    /* Explanation text */
    .why-text {
        font-size: 14px; color: #374151;
        line-height: 1.65;
        margin: 0;
    }

    /* Signal pills */
    .signal-pill {
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
    }
    .signal-positive {background: #DCFCE7; color: #166534;}
    .signal-negative {background: #FEE2E2; color: #991B1B;}
    .signal-neutral  {background: #F3F4F6; color: #374151;}
</style>
""", unsafe_allow_html=True)

# ---------- Load models ----------
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

# ---------- Helpers ----------
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
    cleaned = clean_comment(text)
    analyzer = load_vader()
    vader_label, vader_score = classify_vader(cleaned, analyzer)
    vader_confidence = abs(vader_score)

    if vader_confidence >= 0.5:
        return {
            'comment': text,
            'final_label': vader_label,
            'resolved_by': 'VADER (Layer 1)',
            'vader_label': vader_label,
            'vader_score': round(vader_score, 4),
            'roberta_label': None,
            'roberta_confidence': None,
        }

    roberta = load_roberta()
    r = roberta(cleaned)[0]
    roberta_label = r['label'].capitalize()
    roberta_conf = round(r['score'], 3)

    if roberta_conf >= 0.7:
        final, resolved = roberta_label, 'RoBERTa (Layer 2)'
    else:
        final, resolved = 'Uncertain', 'Flagged for review'

    return {
        'comment': text,
        'final_label': final,
        'resolved_by': resolved,
        'vader_label': vader_label,
        'vader_score': round(vader_score, 4),
        'roberta_label': roberta_label,
        'roberta_confidence': roberta_conf,
    }

# ---------- Session state ----------
if 'results' not in st.session_state:
    st.session_state.results = None

# ---------- Nav bar ----------
st.markdown("""
<div class="nav-bar">
    <div class="nav-left">
        <div class="nav-logo">💬</div>
        <div>
            <p class="nav-title">SentimentAnalyzer</p>
            <p class="nav-subtitle">Social Media Comments Sentiment Analysis</p>
        </div>
    </div>
    <div class="nav-links">
        <span class="nav-link-active">Home</span>
        <span>About</span>
        <span>Contact</span>
        <span style="font-size:16px;">☀</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Main layout ----------
col_left, col_right = st.columns([1.4, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="hero-title">
        Turn Social Media Comments into <span class="accent">Insights</span>
    </div>
    <div class="hero-subtitle">
        Discover what people feel about your brand, product or service.
        Get instant sentiment analysis with easy-to-read results.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">📝 Paste Your Comments Here</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Enter one comment per line (you can also paste multiple comments).</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "comments",
        value=st.session_state.get('input_text', ''),
        height=170,
        placeholder="e.g. I love this product!\nThis service is terrible...",
        label_visibility="collapsed",
    )

    btn_left, btn_right = st.columns([1, 1])
    with btn_left:
        if st.button("📋 Load example comments", use_container_width=True):
            st.session_state['input_text'] = (
                "I love this product! It's amazing!\n"
                "This service is terrible, very disappointed.\n"
                "The product is okay, nothing special.\n"
                "Absolutely the best purchase I've made 🔥🔥"
            )
            st.rerun()
    with btn_right:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked:
        lines = [l.strip() for l in user_text.split('\n') if l.strip()]
        if not lines:
            st.warning("Please enter at least one comment.")
        else:
            with st.spinner("Analyzing comments..."):
                st.session_state.results = [analyze_pipeline(line) for line in lines]

with col_right:
    st.markdown('<div class="section-title">Sentiment Overview</div>', unsafe_allow_html=True)

    results = st.session_state.results or []
    total = len(results)
    pos = sum(1 for r in results if r['final_label'] == 'Positive')
    neu = sum(1 for r in results if r['final_label'] == 'Neutral')
    neg = sum(1 for r in results if r['final_label'] == 'Negative')
    pct = lambda n: f"{(n / total * 100):.0f}%" if total else "0%"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-emoji">😊</div>
            <div class="metric-label">Positive</div>
            <div class="metric-value">{pos}</div>
            <div class="metric-pct">{pct(pos)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-emoji">😐</div>
            <div class="metric-label">Neutral</div>
            <div class="metric-value">{neu}</div>
            <div class="metric-pct">{pct(neu)}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-emoji">😞</div>
            <div class="metric-label">Negative</div>
            <div class="metric-value">{neg}</div>
            <div class="metric-pct">{pct(neg)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <span>💡</span>
        <span>Paste your comments and click Analyze to see the sentiment breakdown, charts and detailed results.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why Use Sentiment Analysis?</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Understand<br>customer opinions</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Improve product<br>and service</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Make data-driven<br>decisions</div>
        </div>""", unsafe_allow_html=True)

# ---------- Helper functions for analysis view ----------
def derive_emotions(compound, pos, neu, neg):
    emotions = {}
    if compound > 0.05:
        intensity = min(abs(compound), 1.0)
        emotions['Love'] = int(min(100, 40 + intensity * 55))
        emotions['Happiness'] = int(min(100, 30 + intensity * 50))
        if pos > 0.5:
            emotions['Satisfaction'] = int(min(100, pos * 95))
    elif compound < -0.05:
        intensity = min(abs(compound), 1.0)
        emotions['Frustration'] = int(min(100, 35 + intensity * 55))
        emotions['Disappointment'] = int(min(100, 25 + intensity * 50))
        if neg > 0.4:
            emotions['Concern'] = int(min(100, neg * 90))
    else:
        emotions['Neutral'] = 70
        emotions['Objective'] = 60
    return emotions

def derive_tone(label, compound):
    if label == 'Positive':
        tones = ['Friendly', 'Appreciative']
        if abs(compound) > 0.7:
            tones.append('Enthusiastic')
        tones.append('Supportive')
    elif label == 'Negative':
        tones = ['Critical', 'Frustrated']
        if abs(compound) > 0.7:
            tones.append('Disapproving')
        tones.append('Candid')
    else:
        tones = ['Objective', 'Neutral']
    return tones

def extract_signals(text, analyzer):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]+\b', text_lower)
    seen = set()
    signals = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if w in analyzer.lexicon:
            score = analyzer.lexicon[w]
            if score >= 1.5:
                signals.append((w, 'positive'))
            elif score <= -1.5:
                signals.append((w, 'negative'))
    return signals[:6]

def generate_explanation(text, label, compound, signals):
    signal_str = ', '.join([s[0] for s in signals[:3]])
    if label == 'Positive':
        base = "The message expresses clear positive sentiment"
        if abs(compound) > 0.7:
            base += " with strong, unambiguous appreciation"
        return f"{base}. Key drivers: {signal_str or 'positive phrasing and tone'}. Concise and direct in its approval."
    elif label == 'Negative':
        base = "The message expresses dissatisfaction"
        if abs(compound) > 0.7:
            base += " with strong, unambiguous criticism"
        return f"{base}. Key drivers: {signal_str or 'negative phrasing and tone'}. Direct in its criticism."
    else:
        return "The message is balanced and objective. No strong emotional charge detected. This reads as neutral, factual, or mixed sentiment."

def render_analysis_view(result, analyzer):
    text = result['comment']
    label = result['final_label']
    cleaned = clean_comment(text)
    scores = analyzer.polarity_scores(cleaned)
    compound = scores['compound']

     # Determine confidence, color, AND marker position consistently
    color_map = {'Positive': '#10b981', 'Neutral': '#6b7280', 'Negative': '#ef4444'}
    color = color_map.get(label, '#6b7280')

    if label == 'Uncertain':
        confidence = 50
        color = '#6b7280'
        marker_pos = 50
    elif result['roberta_confidence'] is not None and result['resolved_by'] == 'RoBERTa (Layer 2)':
        confidence = int(result['roberta_confidence'] * 100)
        # Position marker based on RoBERTa's verdict, not VADER's compound
        if label == 'Positive':
            marker_pos = 50 + (confidence / 2)
        elif label == 'Negative':
            marker_pos = 50 - (confidence / 2)
        else:
            marker_pos = 50
    else:
        confidence = int(min(99, abs(compound) * 100))
        # VADER resolved it — use compound directly
        marker_pos = ((compound + 1) / 2) * 100

    # Row 1
    row1a, row1b = st.columns(2, gap="medium")

    with row1a:
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Overall Sentiment</p>'
            '<div class="donut-wrap">'
            f'<div class="donut" style="background: conic-gradient({color} 0% {confidence}%, #E5E7EB {confidence}% 100%);">'
            '<div class="donut-inner">'
            f'<div class="donut-value">{confidence}%</div>'
            '<div class="donut-label">Confidence</div>'
            '</div>'
            '</div>'
            '<div style="flex:1;">'
            f'<p class="donut-sentiment" style="color:{color};">{label}</p>'
            f'<p class="donut-sub">Sentiment confidence: {confidence}%</p>'
            '<div class="grad-bar">'
            f'<div class="grad-marker" style="left:{marker_pos}%;"></div>'
            '</div>'
            '<div class="grad-labels"><span>Negative</span><span>Positive</span></div>'
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with row1b:
        emotions = derive_emotions(compound, scores['pos'], scores['neu'], scores['neg'])
        primary = max(emotions, key=emotions.get)
        rows_html = ""
        for emo, val in list(emotions.items())[:3]:
            rows_html += (
                f'<div class="emo-row">'
                f'<div class="emo-head"><span>{emo}</span><span>{val}%</span></div>'
                f'<div class="emo-bar"><div class="emo-fill" style="width:{val}%;"></div></div>'
                f'</div>'
            )
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Emotional Tone</p>'
            '<p style="font-size:12px;color:#6B7280;margin:0 0 14px 0;">Primary emotion</p>'
            f'<p style="font-size:22px;font-weight:700;color:#111827;margin:0 0 18px 0;">{primary}</p>'
            f'{rows_html}'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Row 2
    row2a, row2b = st.columns(2, gap="medium")

    with row2a:
        intensity = abs(compound)
        if intensity < 0.3:
            level, active_segs = 'Low intensity', 1
        elif intensity < 0.5:
            level, active_segs = 'Moderate intensity', 2
        elif intensity < 0.7:
            level, active_segs = 'High intensity', 3
        else:
            level, active_segs = 'High intensity', 4
        segs = ''.join([f'<div class="intensity-seg {"active" if i < active_segs else ""}"></div>' for i in range(4)])
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Emotional Intensity</p>'
            f'<p class="intensity-value">{int(intensity * 100)}% <span class="intensity-label">- {level}</span></p>'
            f'<div class="intensity-bar">{segs}</div>'
            '<div class="intensity-labels"><span>Low</span><span>Moderate</span><span>High</span><span>Very High</span></div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with row2b:
        tones = derive_tone(label, compound)
        pills = ''.join([f'<span class="pill">{t}</span>' for t in tones])
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Communication Tone</p>'
            f'<div class="pill-row">{pills}</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Row 3
    row3a, row3b = st.columns(2, gap="medium")

    with row3a:
        signals_for_explanation = extract_signals(cleaned, analyzer)
        explanation = generate_explanation(text, label, compound, signals_for_explanation)
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Why We Think This</p>'
            f'<p class="why-text">{explanation}</p>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with row3b:
        signals = extract_signals(cleaned, analyzer)
        if signals:
            pills = ''.join([f'<span class="signal-pill signal-{s[1]}">{s[0]}</span>' for s in signals])
        else:
            pills = '<span class="signal-pill signal-neutral">No strong signals</span>'
        card_html = (
            '<div class="analysis-card">'
            '<p class="analysis-label">Key Emotional Signals</p>'
            f'<div class="pill-row">{pills}</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

# ---------- Render rich results ----------
if st.session_state.results:
    st.markdown("---")

    results = st.session_state.results
    analyzer = load_vader()

    if len(results) == 1:
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
        render_analysis_view(results[0], analyzer)
    else:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
        with st.expander(f"🔍 Detailed view - {results[0]['comment'][:60]}...", expanded=True):
            render_analysis_view(results[0], analyzer)

        st.markdown('<div style="margin-top:18px;font-size:13px;color:#6B7280;">Other comments:</div>', unsafe_allow_html=True)
        for r in results[1:]:
            label = r['final_label']
            cls = {
                'Positive': 'badge-positive',
                'Neutral': 'badge-neutral',
                'Negative': 'badge-negative',
            }.get(label, 'badge-review')
            st.markdown(f"""<div class="result-row">
                <span class="result-badge {cls}">{label}</span>
                <span class="result-comment">{r['comment']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Download all results as CSV",
        csv,
        "sentiment_results.csv",
        "text/csv",
    )

# ---------- Batch CSV ----------
with st.expander("📁 Batch analysis - upload a CSV"):
    uploaded = st.file_uploader("CSV with a 'text' column", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df = df.dropna(subset=['text'])
            comments = df['text'].astype(str).tolist()
            st.success(f"Loaded {len(comments)} comments")
            if st.button("Run batch analysis"):
                rows = []
                progress = st.progress(0)
                for i, c in enumerate(comments):
                    rows.append(analyze_pipeline(c))
                    progress.progress((i + 1) / len(comments))
                df_out = pd.DataFrame(rows)
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "⬇ Download batch results",
                    df_out.to_csv(index=False).encode('utf-8'),
                    "batch_results.csv",
                    "text/csv",
                )
