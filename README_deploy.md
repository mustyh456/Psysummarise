# PsySummarise — Deployment Guide

## Local run
```bash
pip install streamlit openai
streamlit run app.py
```
Make sure extraction_schema.json is in the same folder as app.py.

## Deploy to Streamlit Community Cloud (free)
1. Push app.py, requirements.txt, extraction_schema.json to a GitHub repo
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file path: app.py
5. Deploy — live URL in ~2 minutes

## Using the app
- Enter your OpenAI API key in the sidebar (never stored)
- Paste any ward round note into the text box
- Click Extract
- Switch between Clinical Summary and Raw JSON tabs
- Download the JSON if needed
