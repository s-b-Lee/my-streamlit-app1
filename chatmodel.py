# app.py
import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="🎨 추구미(이미지 정체성) 설계 AI",
    page_icon="🎨",
    layout="wide",
)

# =========================
# Session State
# =========================
def init_state():
    st.session_state.setdefault("openai_api_key", "")
    st.session_state.setdefault("model", "gpt-4-mini")

    # Inputs
    st.session_state.setdefault("selected_cards", [])
    st.session_state.setdefault("dislikes", "")
    st.session_state.setdefault("wants", "")
    st.session_state.setdefault("constraints", "")
    st.session_state.setdefault("places", [])
    st.session_state.setdefault("notes", "")

    # Outputs
    st.session_state.setdefault("style_report", "")
    st.session_state.setdefault("inspo_analysis", "")
    st.session_state.setdefault("fit_feedback", "")

    # Tracker
    st.session_state.setdefault("style_logs", [])  # list[dict]

    # Memory
    st.session_state.setdefault("last_profile_summary", "")  # short summary for follow-up
