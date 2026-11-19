"""Minimal local Streamlit interface for the trained detector."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.features import load_waveform
from src.predict import load_detector, score_waveform


st.set_page_config(page_title="VoxTrace | Audio Review", page_icon="◉", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f5f7fa; color: #17212b; }
    [data-testid="stHeader"] { background: rgba(245,247,250,.92); }
    .block-container { max-width: 820px; padding-top: 2.6rem; padding-bottom: 3rem; }
    .eyebrow { color: #52707c; font-size: .76rem; font-weight: 700; letter-spacing: .14em; }
    h1 { color: #17212b; letter-spacing: -.04em; font-size: 2.65rem !important; margin: .35rem 0 .3rem !important; }
    .subhead { color: #5b6873; font-size: 1.03rem; line-height: 1.6; margin-bottom: 1.5rem; }
    .notice { background: #eef4f5; border-left: 3px solid #5f9295; border-radius: 8px; padding: .8rem 1rem; color: #3d5960; font-size: .9rem; }
    .result-title { font-size: 1.15rem; font-weight: 700; color: #17212b; }
    .result-tag { color: #52707c; font-size: .77rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .stButton > button[kind="primary"] { background: #176b70; border: 0; border-radius: 10px; min-height: 2.9rem; font-weight: 650; }
    .stButton > button[kind="primary"]:hover { background: #12585c; border: 0; }
    [data-testid="stFileUploader"] { background: #fff; border-radius: 12px; }
    hr { border-color: #e1e7ec; }
    @media (max-width: 640px) { h1 { font-size: 2.1rem !important; } .block-container { padding-top: 1.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector(kind: str):
    cfg = load_config()
    detector = load_detector(kind, cfg)
    return detector[1].get("config", cfg), detector


st.markdown('<div class="eyebrow">VOXTRACE · SPEECH AUTHENTICITY RESEARCH</div>', unsafe_allow_html=True)
st.title("Review an audio clip")
st.markdown(
    '<div class="subhead">A small research tool that scores speech against patterns learned from the training data.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice">Experimental result only. A model score is not proof that a recording is genuine or synthetic.</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.form("audio_review"):
    model_name = st.selectbox("Detector", ["Log-Mel CNN", "MFCC baseline"], help="Train the selected model before analysis.")
    uploaded = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "ogg", "m4a"], label_visibility="visible")
    submitted = st.form_submit_button("Analyze audio", type="primary", use_container_width=True)

kind = "cnn" if model_name == "Log-Mel CNN" else "mfcc"
if uploaded is not None:
    st.audio(uploaded)

if submitted:
    if uploaded is None:
        st.info("Choose an audio file to begin.")
    else:
        try:
            cfg, (model, metadata, device) = get_detector(kind)
            with tempfile.TemporaryDirectory() as folder:
                temp = Path(folder) / f"upload{Path(uploaded.name).suffix}"
                temp.write_bytes(uploaded.getvalue())
                waveform = load_waveform(temp, int(cfg["sample_rate"]))
            score, pieces = score_waveform(waveform, kind, cfg, model, device)
            threshold = float(metadata["threshold"])
            is_spoof = score >= threshold
            verdict = "Synthetic-like patterns detected" if is_spoof else "More consistent with genuine speech"
            st.write("")
            st.markdown('<div class="result-tag">Analysis result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">{verdict}</div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            left.metric("Synthetic score", f"{score:.1%}")
            right.metric("Decision threshold", f"{threshold:.1%}")
            st.progress(min(max(score, 0.0), 1.0), text="Model score (not calibrated certainty)")
            if len(pieces) > 1:
                score_frame = pd.DataFrame(
                    {"Segment score": pieces, "Decision threshold": [threshold] * len(pieces)},
                    index=pd.Index(range(1, len(pieces) + 1), name="Audio segment"),
                )
                st.caption("Score by overlapping segment")
                st.line_chart(score_frame, height=200)
                sides = {piece >= threshold for piece in pieces}
                if len(sides) > 1:
                    st.warning("The segments do not all agree. Treat the overall score cautiously and review the recording.")
            st.caption(f"Audio length: {len(waveform) / int(cfg['sample_rate']):.1f} seconds · {len(pieces)} overlapping segment(s) scored.")
            st.caption("A single score can be wrong, especially for unfamiliar generators or recording conditions.")
        except FileNotFoundError:
            st.error("The trained model files are missing. Follow the training steps in the README, then reopen the app.")
        except Exception as exc:
            st.error(f"This file could not be analyzed: {exc}")

with st.expander("How to read this result"):
    st.write(
        "The synthetic score is the detector's output for the uploaded clip. The threshold was selected on the validation split. "
        "This tool does not establish who spoke, how the audio was made, or whether it was edited. Review the recording and its provenance separately."
    )

st.divider()
st.caption("VoxTrace · Runs locally when launched with `streamlit run app.py` · Audio is not sent to a hosted service by this app.")
