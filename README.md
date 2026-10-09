# Social Media Sentiment Analyzer
title: Social Media Sentiment Analyzer
emoji: 💬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Social Media Sentiment Analyzer

A two-layer sentiment analysis pipeline for social media comments.

## How it works

- **Layer 1 — VADER**: A fast, lexicon-based model that screens every comment instantly. It handles clear cases (strong praise, strong complaints, emoji-only comments).
- **Layer 2 — Twitter-RoBERTa**: A transformer-based model that kicks in only when VADER is uncertain. It handles slang, sarcasm, multilingual text, and context-dependent sentiment.

## Why two layers?

Running a heavy transformer model on every comment is expensive and slow. Most comments are easy — a fast model handles them. Only the ambiguous ones need deep analysis. This mirrors how production sentiment systems work at scale: cheap models handle the easy 70%, expensive models handle the hard 30%.

## Features

- **Single Comment tab** — paste a comment and see both layers in action
- **Batch Analysis tab** — upload a CSV with a `text` column and analyze hundreds of comments at once
- **Routing visualizations** — see what percentage of comments were resolved by each layer

## Tech stack

- [Streamlit](https://streamlit.io) — web UI
- [VADER](https://github.com/cjhutto/vaderSentiment) — lexicon-based sentiment
- [Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) — transformer-based sentiment
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — model loading

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
