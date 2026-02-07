# app.py
import base64
import datetime as dt
import json
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="통합 AI 앱 (상담사 + 감정 트래커 + 추구미 설계)",
    page_icon="🧠✨",
    layout="wide",
)

# -----------------------------
# Constants / Config
# -----------------------------
OPENAI_MODEL = "gpt-4-mini"
PINTEREST_BASE = "https://api.pinterest.com/v5"

MOOD_CHOICES = [
    ("😄", "좋음"),
    ("🙂", "괜찮음"),
    ("😐", "보통"),
    ("😟", "불안"),
    ("😢", "슬픔"),
    ("😠", "분노"),
    ("🥱", "지침"),
    ("✨", "설렘"),
]

EMOTION_LABELS = ["슬픔", "불안", "분노", "지침", "허무", "설렘", "외로움", "긴장", "무기력", "기대", "안도", "복잡함"]

STYLE_KEYWORDS = [
    "세련됨", "우아함", "여성스러움", "중성적인", "절제된", "귀여움", "청순함", "강렬한",
    "섹시한", "무채색의", "시크함", "고급스러움", "러블리", "단아한", "단정한",
]

SPACE_CHOICES = ["학교", "직장", "데이트", "SNS", "공식 자리"]

PERSONAS = {
    "친한 친구": "친근하고 따뜻하되 과장하지 말고, 편하게 말하되 해결로 이어지게.",
    "차분한 전문가": "차분하고 안정적이며 구조적으로 정리해서 말하기.",
    "코치 스타일": "목표-현실-옵션-실행으로 이끄는 코칭 톤, 단 실행 가능하게.",
}

CATEGORIES = ["자기계발", "커리어", "연애", "인간관계", "기타"]

PRIVACY_NOTICE = (
    "⚠️ **고지**: 이 앱은 의료/심리 **진단**을 제공하지 않습니다. "
    "자해/자살 등 위기 상황이 있거나 안전이 우려되면, 즉시 112/119 또는 "
    "가까운 응급실/전문기관의 도움을 받으세요."
)

# Pinterest Search Notes (important to set expectations)
PINTEREST_NOTE = (
    "ℹ️ Pinterest API는 **OAuth Access Token(베어러 토큰)** 기반입니다. "
    "또한 `GET /v5/search/partner/pins`는 **베타이며 모든 앱에서 사용 불가**일 수 있어요. "
    "사용 불가(403 등)면 앱에서 안내 문구가 표시됩니다."
)

# -----------------------------
# Session State Init
# -----------------------------
def init_state():
    defaults = {
        "messages": [],  # 상담 대화
        "turn_count": 0,
        "mood_logs": [],  # 감정 기록
        "persona": "차분한 전문가",
        "category": "자기계발",
        "move_to_style": False,
        "counsel_summary_for_style": "",
        "style_inputs": {
            "keywords": [],
            "text_like": "",
            "text_dislike": "",
            "text_constraints": "",
            "spaces": [],
            "uploaded_image_bytes": None,
            "uploaded_image_name": None,
            "uploaded_image_analysis": None,
        },
        "style_report": None,
        "last_emotion_guess": None,
        "last_emotion_guess_reason": None,
        "pinterest_cache": {},  # term -> pins list
        "pinterest_last_term": "",
        "active_tab": 0,  # 0 상담, 1 트래커, 2 추구미
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# -----------------------------
# Helpers: Safety / Signals
# -----------------------------
CRISIS_PATTERNS = [
    r"자살", r"죽고\s*싶", r"죽고싶", r"자해", r"해치고\s*싶", r"목숨", r"극단적\s*선택",
    r"살\s*기\s*싫", r"사라지고\s*싶",
]

STYLE_SIGNAL_PATTERNS = [
    r"이미지", r"분위기", r"정체성", r"첫인상", r"스타일", r"외모", r"옷", r"메이크업",
    r"꾸미", r"브랜딩", r"인상", r"자신감.*외모", r"자신감.*스타일",
]


def detect_crisis(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(p, t) for p in CRISIS_PATTERNS)


def detect_style_signal(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(p, t) for p in STYLE_SIGNAL_PATTERNS)


# -----------------------------
# OpenAI (Streaming) via REST
# -----------------------------
def openai_stream_chat(
    api_key: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
) -> str:
    """
    Stream response safely using a single placeholder (st.empty).
    Uses OpenAI Chat Completions-compatible REST path.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "stream": True,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }

    placeholder = st.empty()
    full_text = ""

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
            if r.status_code != 200:
                try:
                    err = r.json()
                except Exception:
                    err = {"error": {"message": r.text}}
                raise RuntimeError(err.get("error", {}).get("message", f"HTTP {r.status_code}"))

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        delta = j["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_text += delta
                            placeholder.markdown(full_text)
                    except Exception:
                        # ignore malformed chunks
                        continue
    except requests.exceptions.Timeout:
        raise RuntimeError("요청 시간이 초과됐어요. 네트워크 상태를 확인하고 다시 시도해 주세요.")
    except requests.exceptions.RequestException:
        raise RuntimeError("네트워크 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")

    return full_text


def openai_json(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": {"message": r.text}}
        raise RuntimeError(err.get("error", {}).get("message", f"HTTP {r.status_code}"))
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def openai_vision_analyze_style(
    api_key: str,
    image_bytes: bytes,
    allowed_keywords: List[str],
) -> Dict[str, Any]:
    """
    Analyze uploaded image for '추구미' cues using a vision-capable chat request.
    Returns JSON: {keywords:[], rationale:"", warnings:""}
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = (
        "당신은 '추구미(이미지 정체성)' 분석가입니다. "
        "사용자가 업로드한 이미지를 보고, 주어진 키워드 후보 중에서만 "
        "이미지의 분위기/스타일에 해당하는 키워드를 골라주세요. "
        "과장하지 말고, 보이는 근거를 짧게 설명하세요. "
        "개인 식별(누구인지, 나이 추정 등)은 하지 마세요. "
        "반드시 JSON으로만 답하세요."
    )

    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"허용 키워드 후보:\n{allowed_keywords}\n\n"
                    "요청:\n"
                    "1) 후보 중 3~7개 키워드를 선택\n"
                    "2) 근거를 한 단락으로 짧게\n"
                    "3) 이미지가 추구미 분석에 부적절/애매하면 경고문(warnings)에 한 줄\n\n"
                    "출력 JSON 스키마:\n"
                    '{ "keywords": [...], "rationale": "...", "warnings": "..." }'
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            user_prompt,
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": {"message": r.text}}
        raise RuntimeError(err.get("error", {}).get("message", f"HTTP {r.status_code}"))
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


# -----------------------------
# Pinterest API helpers
# -----------------------------
def pinterest_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def pinterest_best_image_url(media: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    PinMediaWithImage.images includes keys like '1200x', '600x', '400x300', '150x150'
    """
    if not media:
        return None
    images = None
    if isinstance(media, dict):
        # For SummaryPin: media is PinMedia, 'images' lives under media when media_type == 'image' or 'video'
        images = media.get("images")
    if not isinstance(images, dict):
        return None
    for key in ["600x", "400x300", "1200x", "150x150"]:
        if key in images and isinstance(images[key], dict) and images[key].get("url"):
            return images[key]["url"]
    # fallback: any dict with url
    for v in images.values():
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
    return None


def pinterest_search_partner_pins(
    access_token: str,
    term: str,
    country_code: str = "KR",
    locale: str = "ko-KR",
    limit: int = 12,
    bookmark: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GET /v5/search/partner/pins (beta; might be unavailable) :contentReference[oaicite:0]{index=0}
    """
    url = f"{PINTEREST_BASE}/search/partner/pins"
    params = {
        "term": term,
        "country_code": country_code,
        "locale": locale,
        "limit": max(1, min(limit, 50)),
    }
    if bookmark:
        params["bookmark"] = bookmark

    r = requests.get(url, headers=pinterest_headers(access_token), params=params, timeout=30)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Pinterest API 오류 ({r.status_code}): {err}")
    return r.json()


def pinterest_terms_suggested(
    access_token: str,
    term: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    GET /v5/terms/suggested (ads:read scope in spec; but can be used if permitted) :contentReference[oaicite:1]{index=1}
    """
    url = f"{PINTEREST_BASE}/terms/suggested"
    params = {"term": term, "limit": max(1, min(limit, 50))}
    r = requests.get(url, headers=pinterest_headers(access_token), params=params, timeout=30)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Pinterest terms 오류 ({r.status_code}): {err}")
    return r.json()


# -----------------------------
# Prompt builders
# -----------------------------
def counselor_system_prompt(category: str, persona: str) -> str:
    return f"""
당신은 대학생/대학원생 대상의 AI 상담사 겸 코치입니다.

말투/성격:
- 두괄식, 필요한 말만, 논리적
- 이해를 돕는 비유는 최대 1회만
- "즉시 공감 + 구체적 행동 제안" 패턴을 기본으로
- 단정하지 말고, 일반적으로 알려진 수준으로 말하되 과장 금지

카테고리: {category}
대화 톤(캐릭터): {persona} ({PERSONAS.get(persona, "")})

안전:
- 자해/자살/위험 신호가 감지되면: 즉시 안전 안내 + 전문기관 권유를 하고,
  안전 확인 질문은 1개만 한다.

주기 요약:
- 6~8턴마다 "요약 + 다음 행동 2~3개"를 짧게 제공한다.

출력 형식:
- 항상 한국어
- 1) 공감 한 문장
- 2) 상황 정리(핵심 1~2문장)
- 3) 다음 행동 제안 2~3개(불릿)
- 필요할 때만 질문 1개
""".strip()


def emotion_label_prompt(user_text: str) -> Tuple[str, str]:
    system_prompt = (
        "당신은 감정 라벨러입니다. 사용자의 문장을 읽고 가장 강한 감정 1개와 보조 감정 1개를 고르세요. "
        "추측임을 명확히 하고, 근거는 짧게. 반드시 JSON으로만 답하세요."
    )
    user_prompt = (
        f"문장:\n{user_text}\n\n"
        f"가능 라벨:\n{EMOTION_LABELS}\n\n"
        'JSON 스키마: {"primary":"", "secondary":"", "reason":"", "trigger_keywords":[...]}'
    )
    return system_prompt, user_prompt


def summarize_for_style_prompt(conversation: List[Dict[str, str]]) -> Tuple[str, str]:
    system_prompt = (
        "당신은 상담 내용을 '추구미 설계'로 넘기기 위한 요약가입니다. "
        "상담 전체에서 핵심 감정/상황/원하는 변화/제약을 뽑아 5~8줄로 요약하세요. "
        "반드시 JSON으로만 답하세요."
    )
    convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation][-20:])
    user_prompt = (
        f"상담 대화(최근 20개):\n{convo_text}\n\n"
        'JSON 스키마: {"core_emotions":[...], "situation":"", "desired_change":"", "constraints":"", "keywords":[...]}'
    )
    return system_prompt, user_prompt


def style_report_prompt(
    style_inputs: Dict[str, Any],
    counselor_summary: str,
) -> Tuple[str, str]:
    system_prompt = (
        "당신은 '추구미 도우미'입니다. "
        "사용자의 선택 키워드/텍스트/상황을 바탕으로 추구미 리포트와 실천 가이드를 생성하세요. "
        "브랜드/제품 추천 금지(방향성만). "
        "과장하지 말고 구조적으로. 반드시 JSON으로만 답하세요."
    )

    user_prompt = {
        "selected_keywords": style_inputs.get("keywords", []),
        "text_like": style_inputs.get("text_like", ""),
        "text_dislike": style_inputs.get("text_dislike", ""),
        "text_constraints": style_inputs.get("text_constraints", ""),
        "spaces": style_inputs.get("spaces", []),
        "uploaded_image_analysis": style_inputs.get("uploaded_image_analysis"),
        "counselor_summary": counselor_summary,
        "output_schema": {
            "type_name_ko": "",
            "type_name_en": "",
            "identity_one_liner": "",
            "core_keywords": [],
            "mini_report": {
                "mood_summary": "",
                "impression": "",
                "best_contexts": [],
                "watch_out": "",
                "maintenance_difficulty": "낮음/중간/높음 중 하나",
            },
            "apply_strategy_from_counseling": "",
            "practice_guide": {
                "makeup": {
                    "base": "",
                    "points": {"eyes": "", "lips": ""},
                    "avoid": "",
                },
                "fashion": {
                    "silhouette": "",
                    "color_palette": [],
                    "avoid_colors": [],
                    "top5_items": [],
                },
                "behavior_lifestyle": {
                    "gesture_tone": "",
                    "speech_manner": "",
                    "daily_habits": [],
                },
            },
        },
    }

    return system_prompt, json.dumps(user_prompt, ensure_ascii=False)


def pinterest_query_expander_prompt(
    chosen_keywords: List[str],
    spaces: List[str],
    locale_hint: str = "Korean",
) -> Tuple[str, str]:
    system_prompt = (
        "당신은 Pinterest 검색어 설계자입니다. "
        "사용자가 선택한 추구미 키워드로 '사람(인물) 이미지'가 잘 나오는 검색어를 만든다. "
        "Pinterest 검색에 강한 짧은 쿼리로 3~6개를 제안하라. "
        "한국어/영어 혼합 가능. "
        "반드시 JSON으로만 답하세요."
    )
    user_prompt = (
        f"키워드: {chosen_keywords}\n"
        f"적용 공간: {spaces}\n"
        f"언어 힌트: {locale_hint}\n\n"
        'JSON 스키마: {"queries":[...], "negative_terms":[...], "note":"..."}\n'
        "- queries는 3~6개, 각 2~6단어\n"
        "- 사람/패션/룩/메이크업 중심(예: 'neutral chic outfit', 'clean girl makeup')"
    )
    return system_prompt, user_prompt


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", value="")
    pinterest_token = st.text_input("Pinterest Access Token (Bearer)", type="password", value="")
    st.caption(PINTEREST_NOTE)

    st.divider()

    st.session_state["category"] = st.selectbox("상담/코칭 카테고리", CATEGORIES, index=CATEGORIES.index(st.session_state["category"]))
    st.session_state["persona"] = st.selectbox("대화 톤", list(PERSONAS.keys()), index=list(PERSONAS.keys()).index(st.session_state["persona"]))

    if st.button("🧹 대화 초기화", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["turn_count"] = 0
        st.session_state["move_to_style"] = False
        st.session_state["counsel_summary_for_style"] = ""
        st.session_state["last_emotion_guess"] = None
        st.session_state["last_emotion_guess_reason"] = None
        st.success("초기화 완료!")

    st.divider()
    st.markdown(PRIVACY_NOTICE)

# -----------------------------
# Tabs with controlled navigation
# -----------------------------
tab_titles = ["🧠 AI 상담사", "📊 감정 트래커", "✨ 추구미 설계"]
tabs = st.tabs(tab_titles)

# -----------------------------
# TAB 1: Counselor Chat
# -----------------------------
with tabs[0]:
    st.title("🧠 AI 상담사")
    st.caption("즉시 공감 + 구체적 행동 제안. 필요하면 자연스럽게 추구미 설계로 연결해요.")

    # render messages
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("지금 어떤 고민이 있나요? (자유롭게 적어주세요)")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["turn_count"] += 1
        with st.chat_message("user"):
            st.markdown(user_input)

        # crisis handling (no model call)
        if detect_crisis(user_input):
            with st.chat_message("assistant"):
                st.markdown(
                    "지금 안전이 가장 중요해요.\n\n"
                    "- **즉시 112/119** 또는 가까운 응급실/전문기관에 도움을 요청해 주세요.\n"
                    "- 주변에 믿을 수 있는 사람(가족/친구/담당자)에게 **지금 곁에 있어달라고** 말해 주세요.\n\n"
                    "한 가지만 확인할게요: **지금 혼자 있나요, 아니면 누군가 곁에 있나요?**"
                )
            st.session_state["messages"].append(
                {"role": "assistant", "content": "지금 안전이 가장 중요해요... (안전 안내 및 확인 질문)"}  # minimal log
            )
        else:
            # emotion label (json)
            if openai_key:
                try:
                    sp, up = emotion_label_prompt(user_input)
                    emo = openai_json(openai_key, sp, up, temperature=0.0)
                    st.session_state["last_emotion_guess"] = emo.get("primary")
                    st.session_state["last_emotion_guess_reason"] = emo.get("reason", "")
                except Exception:
                    st.session_state["last_emotion_guess"] = None
                    st.session_state["last_emotion_guess_reason"] = None

            # counselor response
            with st.chat_message("assistant"):
                if not openai_key:
                    st.warning("사이드바에 OpenAI API Key를 입력하면 상담 응답을 받을 수 있어요.")
                else:
                    try:
                        sys_p = counselor_system_prompt(st.session_state["category"], st.session_state["persona"])
                        assistant_text = openai_stream_chat(openai_key, sys_p, st.session_state["messages"], temperature=0.7)
                        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

                        # periodically summarize
                        if st.session_state["turn_count"] % 7 == 0:
                            try:
                                sp2, up2 = summarize_for_style_prompt(st.session_state["messages"])
                                summ = openai_json(openai_key, sp2, up2, temperature=0.2)
                                summary_lines = [
                                    f"- 핵심 감정: {', '.join(summ.get('core_emotions', [])[:3])}",
                                    f"- 상황: {summ.get('situation','')}",
                                    f"- 원하는 변화: {summ.get('desired_change','')}",
                                    f"- 제약/현실: {summ.get('constraints','')}",
                                ]
                                st.markdown("#### 🧾 중간 요약")
                                st.markdown("\n".join(summary_lines))
                            except Exception:
                                pass

                    except Exception as e:
                        st.error(f"오류가 발생했어요: {e}")

            # emotion quick save button
            if st.session_state["last_emotion_guess"]:
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    if st.button("📌 오늘 감정으로 저장", use_container_width=True):
                        now = dt.datetime.now()
                        st.session_state["mood_logs"].append(
                            {
                                "ts": now.isoformat(timespec="seconds"),
                                "date": now.date().isoformat(),
                                "weekday": now.strftime("%a"),
                                "mood": "😐",
                                "mood_name": "보통",
                                "memo": user_input[:200],
                                "label": st.session_state["last_emotion_guess"],
                            }
                        )
                        st.success("감정 트래커에 저장했어요!")
                with col_b:
                    st.caption(f"추정 감정: **{st.session_state['last_emotion_guess']}** · {st.session_state['last_emotion_guess_reason'] or ''}")

            # style-signal detection => propose transition
            if detect_style_signal(user_input):
                st.session_state["move_to_style"] = True

            if st.session_state["move_to_style"] and openai_key:
                # build counselor summary for tab3
                if not st.session_state["counsel_summary_for_style"]:
                    try:
                        sp3, up3 = summarize_for_style_prompt(st.session_state["messages"])
                        summ2 = openai_json(openai_key, sp3, up3, temperature=0.2)
                        st.session_state["counsel_summary_for_style"] = (
                            "핵심 감정: " + ", ".join(summ2.get("core_emotions", [])[:3]) + "\n"
                            "상황: " + (summ2.get("situation", "") or "") + "\n"
                            "원하는 변화: " + (summ2.get("desired_change", "") or "") + "\n"
                            "제약: " + (summ2.get("constraints", "") or "")
                        )
                    except Exception:
                        st.session_state["counsel_summary_for_style"] = ""

                st.info("추구미(이미지 정체성) 쪽으로 이어가도 괜찮을까요?")
                if st.button("✨ 추구미 설계 시작", use_container_width=True):
                    st.session_state["active_tab"] = 2
                    st.rerun()

# -----------------------------
# TAB 2: Mood Tracker
# -----------------------------
with tabs[1]:
    st.title("📊 감정 트래커")
    st.caption("오늘 기분을 기록하고, 패턴을 가볍게 확인해요.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 오늘 기록")
        mood_emoji = st.selectbox("기분(이모지)", [m[0] for m in MOOD_CHOICES], index=2)
        mood_name = dict(MOOD_CHOICES).get(mood_emoji, "보통")
        memo = st.text_area("짧은 메모", placeholder="무슨 일이 있었나요? (선택)", height=120)
        label = st.selectbox("감정 라벨(선택)", ["(자동/미선택)"] + EMOTION_LABELS, index=0)

        if st.button("✅ 저장", use_container_width=True):
            now = dt.datetime.now()
            st.session_state["mood_logs"].append(
                {
                    "ts": now.isoformat(timespec="seconds"),
                    "date": now.date().isoformat(),
                    "weekday": now.strftime("%a"),
                    "mood": mood_emoji,
                    "mood_name": mood_name,
                    "memo": (memo or "").strip()[:400],
                    "label": "" if label == "(자동/미선택)" else label,
                }
            )
            st.success("저장했어요!")

        st.divider()
        st.markdown("🧘 마음 안정 콘텐츠(간단)")
        if st.button("🌬️ 60초 호흡 가이드", use_container_width=True):
            st.markdown(
                "- 4초 들이마시기\n"
                "- 4초 멈추기\n"
                "- 6초 내쉬기\n"
                "- 2초 멈추기\n\n"
                "이 사이클을 5번 반복해 보세요."
            )

    with col2:
        st.subheader("📚 기록 목록")
        if not st.session_state["mood_logs"]:
            st.info("아직 기록이 없어요. 왼쪽에서 저장해 보세요.")
        else:
            df = pd.DataFrame(st.session_state["mood_logs"])
            df_show = df[["date", "weekday", "mood", "mood_name", "label", "memo"]].copy()
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            st.subheader("📈 요일별 기분 분포(간단)")
            mood_rank = {name: i for i, name in enumerate(["슬픔", "불안", "분노", "지침", "허무", "보통", "괜찮음", "좋음", "설렘"], start=1)}
            # use mood_name as proxy score
            df_score = df.copy()
            df_score["score"] = df_score["mood_name"].map({"슬픔": 2, "불안": 3, "분노": 3, "지침": 3, "보통": 5, "괜찮음": 6, "좋음": 7, "설렘": 8}).fillna(5)
            order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            df_score["weekday"] = pd.Categorical(df_score["weekday"], categories=order, ordered=True)

            chart = (
                alt.Chart(df_score)
                .mark_bar()
                .encode(
                    x=alt.X("weekday:N", title="요일"),
                    y=alt.Y("mean(score):Q", title="평균 기분(대략)"),
                    tooltip=["weekday", alt.Tooltip("mean(score):Q", title="평균")],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

            st.subheader("🔎 인사이트(키워드 요약)")
            text_blob = " ".join([str(x) for x in df["memo"].tolist() if x])
            tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", text_blob)
            common = [w for w, c in Counter(tokens).most_common(10)]
            if common:
                st.markdown("자주 등장한 단어: " + ", ".join([f"`{w}`" for w in common]))
                st.caption("반복적으로 힘들다면(예: 특정 주기/상황), 전문가 상담을 **가능성**으로 고려해도 좋아요. (진단은 불가)")
            else:
                st.caption("메모가 쌓이면 키워드 인사이트가 더 잘 보여요.")

# -----------------------------
# TAB 3: Style Identity ("추구미") + Pinterest + Image analysis
# -----------------------------
with tabs[2]:
    st.title("✨ 추구미 도우미 - 당신을 브랜딩하는 첫걸음, 추구미")
    st.caption("선택 키워드 + 텍스트 + (선택) 이미지로 추구미 리포트를 만들고, Pinterest 이미지 참고도 붙여요.")

    # Auto-inject counseling summary if moved
    if st.session_state.get("counsel_summary_for_style"):
        st.info("✅ 상담 탭의 요약이 자동 전달됐어요.")
        st.text_area(
            "상담 요약(자동)",
            value=st.session_state["counsel_summary_for_style"],
            height=110,
            disabled=True,
        )

    st.subheader("1) 무드/스타일 선택 (5~10개)")
    selected = st.multiselect(
        "끌리는 키워드를 골라주세요",
        STYLE_KEYWORDS,
        default=st.session_state["style_inputs"].get("keywords", []),
        max_selections=10,
    )
    st.session_state["style_inputs"]["keywords"] = selected

    st.subheader("2) 텍스트 보조 입력")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.session_state["style_inputs"]["text_like"] = st.text_area(
            "내가 좋아하는 스타일을 구체적으로 적어보아요.",
            value=st.session_state["style_inputs"].get("text_like", ""),
            placeholder="예: 편해 보이는데 세련됐으면 / 피부 표현은 깨끗하게",
            height=120,
        )
    with col_b:
        st.session_state["style_inputs"]["text_dislike"] = st.text_area(
            "이런 느낌은 싫어요",
            value=st.session_state["style_inputs"].get("text_dislike", ""),
            placeholder="예: 너무 꾸민 느낌 / 과한 펄",
            height=120,
        )
    with col_c:
        st.session_state["style_inputs"]["text_constraints"] = st.text_area(
            "현실 제약/조건(선택)",
            value=st.session_state["style_inputs"].get("text_constraints", ""),
            placeholder="예: 학교에서 무난해야 함 / 예산 제한 / 관리 시간 적음",
            height=120,
        )

    st.subheader("3) (선택) 사진 업로드 — 추구미 분위기 분석")
    up = st.file_uploader("좋다고 느꼈던 이미지가 있으면 올려주세요 (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        img_bytes = up.read()
        st.session_state["style_inputs"]["uploaded_image_bytes"] = img_bytes
        st.session_state["style_inputs"]["uploaded_image_name"] = up.name
        st.image(img_bytes, caption=f"업로드: {up.name}", use_container_width=True)

        if st.button("🧠 업로드 이미지로 추구미 키워드 추정", use_container_width=True):
            if not openai_key:
                st.warning("OpenAI API Key를 입력하면 이미지 분석을 할 수 있어요.")
            else:
                with st.spinner("이미지 분위기를 분석 중..."):
                    try:
                        analysis = openai_vision_analyze_style(openai_key, img_bytes, STYLE_KEYWORDS)
                        st.session_state["style_inputs"]["uploaded_image_analysis"] = analysis
                        st.success("이미지 기반 키워드 추정 완료!")
                    except Exception as e:
                        st.error(f"이미지 분석 오류: {e}")

    if st.session_state["style_inputs"].get("uploaded_image_analysis"):
        a = st.session_state["style_inputs"]["uploaded_image_analysis"]
        st.markdown("#### 🖼️ 이미지 분석 결과(참고)")
        st.markdown(f"- 추정 키워드: **{', '.join(a.get('keywords', []))}**")
        if a.get("rationale"):
            st.caption(a["rationale"])
        if a.get("warnings"):
            st.warning(a["warnings"])

        if st.button("➕ 이미지 키워드를 선택 키워드에 합치기", use_container_width=True):
            merged = list(dict.fromkeys(st.session_state["style_inputs"]["keywords"] + a.get("keywords", [])))
            st.session_state["style_inputs"]["keywords"] = merged[:10]
            st.rerun()

    st.subheader("4) 적용 공간 선택")
    spaces = st.multiselect(
        "어떤 공간/상황에서 이 추구미를 주로 쓰고 싶나요?",
        SPACE_CHOICES,
        default=st.session_state["style_inputs"].get("spaces", []),
    )
    st.session_state["style_inputs"]["spaces"] = spaces

    st.divider()

    # Pinterest integration
    st.subheader("🧷 Pinterest 참고 이미지(인물 이미지 검색)")
    st.caption("선택한 추구미 키워드로 Pinterest에서 참고 이미지를 가져옵니다(권한/토큰 필요).")

    if not pinterest_token:
        st.info("사이드바에 Pinterest Access Token을 입력하면 Pinterest 이미지를 붙일 수 있어요.")
    else:
        colp1, colp2 = st.columns([2, 1])
        with colp1:
            manual_term = st.text_input("직접 검색어(선택)", value=st.session_state.get("pinterest_last_term", ""))
        with colp2:
            st.write("")
            st.write("")
            auto_expand = st.checkbox("🤖 AI로 검색어 추천", value=True)

        suggested_queries = []
        negative_terms = []
        if auto_expand and openai_key and st.session_state["style_inputs"]["keywords"]:
            if st.button("🔎 검색어 추천 만들기", use_container_width=True):
                try:
                    spx, upx = pinterest_query_expander_prompt(
                        st.session_state["style_inputs"]["keywords"],
                        st.session_state["style_inputs"]["spaces"],
                        locale_hint="Korean + English mix",
                    )
                    qq = openai_json(openai_key, spx, upx, temperature=0.2)
                    suggested_queries = qq.get("queries", [])[:6]
                    negative_terms = qq.get("negative_terms", [])[:6]
                    st.session_state["pinterest_suggested_queries"] = suggested_queries
                    st.session_state["pinterest_negative_terms"] = negative_terms
                except Exception as e:
                    st.error(f"검색어 추천 오류: {e}")

        suggested_queries = st.session_state.get("pinterest_suggested_queries", []) or suggested_queries
        negative_terms = st.session_state.get("pinterest_negative_terms", []) or negative_terms

        if suggested_queries:
            st.markdown("**추천 검색어:** " + " · ".join([f"`{q}`" for q in suggested_queries]))
        if negative_terms:
            st.caption("제외(참고): " + ", ".join([f"`{q}`" for q in negative_terms]))

        term_to_search = manual_term.strip()
        if not term_to_search and suggested_queries:
            term_to_search = suggested_queries[0]

        cols_btn = st.columns([1, 1, 2])
        with cols_btn[0]:
            do_search = st.button("📌 Pinterest 검색", use_container_width=True)
        with cols_btn[1]:
            clear_cache = st.button("🧽 Pinterest 캐시 비우기", use_container_width=True)
        with cols_btn[2]:
            st.caption("※ /search/partner/pins는 베타라 403이면 사용 불가 안내가 나옵니다.")

        if clear_cache:
            st.session_state["pinterest_cache"] = {}
            st.success("캐시를 비웠어요!")

        pins = []
        if do_search:
            if not term_to_search:
                st.warning("검색어를 입력하거나(또는 추천 검색어 생성) 진행해 주세요.")
            else:
                st.session_state["pinterest_last_term"] = term_to_search
                cache = st.session_state["pinterest_cache"]
                if term_to_search in cache:
                    pins = cache[term_to_search]
                else:
                    with st.spinner("Pinterest에서 핀을 불러오는 중..."):
                        try:
                            data = pinterest_search_partner_pins(
                                pinterest_token,
                                term_to_search,
                                country_code="KR",
                                locale="ko-KR",
                                limit=12,
                            )
                            items = data.get("items", []) or []
                            # normalize minimal fields
                            norm = []
                            for it in items:
                                media = it.get("media") or {}
                                img_url = pinterest_best_image_url(media)
                                norm.append(
                                    {
                                        "id": it.get("id"),
                                        "title": it.get("title") or "",
                                        "description": it.get("description") or "",
                                        "link": it.get("link") or "",
                                        "img": img_url,
                                        "alt_text": it.get("alt_text") or "",
                                    }
                                )
                            pins = norm
                            cache[term_to_search] = pins
                            st.session_state["pinterest_cache"] = cache
                        except Exception as e:
                            st.error(
                                "Pinterest API에서 핀을 가져오지 못했어요.\n\n"
                                f"- 사유: {e}\n\n"
                                "가능한 원인:\n"
                                "- 이 앱/토큰이 `GET /v5/search/partner/pins`(베타) 권한이 없음\n"
                                "- 토큰 만료/스코프 부족\n"
                                "- 레이트리밋/네트워크\n"
                            )

        if not pins and term_to_search in st.session_state["pinterest_cache"]:
            pins = st.session_state["pinterest_cache"][term_to_search]

        if pins:
            st.markdown(f"#### 결과: `{term_to_search}`")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for i, p in enumerate(pins):
                with cols[i % 3]:
                    if p.get("img"):
                        # clickable image via HTML
                        link = p.get("link") or "https://www.pinterest.com/"
                        title = (p.get("title") or "").strip() or "Pinterest Pin"
                        st.markdown(
                            f"""
                            <a href="{link}" target="_blank" style="text-decoration:none;">
                                <img src="{p["img"]}" style="width:100%; border-radius:14px; margin-bottom:6px;" />
                            </a>
                            <div style="font-weight:700; margin-bottom:8px;">{title}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("이미지 URL이 없는 핀이에요.")
                    with st.expander("상세"):
                        if p.get("description"):
                            st.write(p["description"])
                        if p.get("alt_text"):
                            st.caption(p["alt_text"])
                        if p.get("link"):
                            st.link_button("Pinterest에서 열기", p["link"])

    st.divider()

    # Generate style report
    st.subheader("🧾 추구미 분석 & 리포트")
    can_run = len(st.session_state["style_inputs"]["keywords"]) >= 5 and len(st.session_state["style_inputs"]["keywords"]) <= 10

    colr1, colr2 = st.columns([1, 2])
    with colr1:
        if st.button("✨ 추구미 분석", use_container_width=True, disabled=not can_run):
            if not openai_key:
                st.warning("OpenAI API Key를 입력해 주세요.")
            else:
                with st.spinner("추구미 리포트를 생성 중..."):
                    try:
                        sys_p, user_p = style_report_prompt(
                            st.session_state["style_inputs"],
                            st.session_state.get("counsel_summary_for_style", ""),
                        )
                        report = openai_json(openai_key, sys_p, user_p, temperature=0.4)
                        st.session_state["style_report"] = report
                        st.success("리포트 생성 완료!")
                    except Exception as e:
                        st.error(f"리포트 생성 오류: {e}")

        st.caption("조건: 키워드 5~10개 선택")
    with colr2:
        st.caption("※ 사진 업로드가 있어도, 현재는 '이미지 내용' 자체를 저장/추적하지 않고 분석 결과(키워드/근거)만 참고합니다.")

    if st.session_state.get("style_report"):
        r = st.session_state["style_report"]
        st.markdown(f"## 💎 타입: **{r.get('type_name_ko','')}**  \n**{r.get('type_name_en','')}**")
        st.markdown(f"**한 문장 정체성:** {r.get('identity_one_liner','')}")
        st.markdown("**핵심 키워드:** " + ", ".join([f"`{k}`" for k in (r.get("core_keywords") or [])]))

        if st.session_state.get("counsel_summary_for_style") and r.get("apply_strategy_from_counseling"):
            st.markdown("### 🧩 현재 고민을 반영한 적용 전략")
            st.write(r["apply_strategy_from_counseling"])

        st.markdown("### 📌 미니 리포트")
        mini = r.get("mini_report", {}) or {}
        st.markdown(f"- 분위기 요약: {mini.get('mood_summary','')}")
        st.markdown(f"- 타인 인상: {mini.get('impression','')}")
        if mini.get("best_contexts"):
            st.markdown("- 어울리는 상황: " + ", ".join([f"`{x}`" for x in mini.get("best_contexts", [])]))
        st.markdown(f"- 과도함 주의: {mini.get('watch_out','')}")
        st.markdown(f"- 유지 난이도: **{mini.get('maintenance_difficulty','')}**")

        st.markdown("### 🪞 실천 가이드 (방향성)")
        guide = r.get("practice_guide", {}) or {}

        m = guide.get("makeup", {}) or {}
        f = guide.get("fashion", {}) or {}
        b = guide.get("behavior_lifestyle", {}) or {}

        cga, cgb = st.columns(2)
        with cga:
            st.markdown("#### 💄 메이크업")
            st.markdown(f"- 베이스: {m.get('base','')}")
            pts = m.get("points", {}) or {}
            st.markdown(f"- 눈: {pts.get('eyes','')}")
            st.markdown(f"- 입술: {pts.get('lips','')}")
            st.markdown(f"- 피하면 좋은 요소: {m.get('avoid','')}")
        with cgb:
            st.markdown("#### 👗 패션")
            st.markdown(f"- 실루엣: {f.get('silhouette','')}")
            if f.get("color_palette"):
                st.markdown("- 컬러 팔레트: " + ", ".join([f"`{x}`" for x in f.get("color_palette", [])]))
            if f.get("avoid_colors"):
                st.markdown("- 피할 컬러: " + ", ".join([f"`{x}`" for x in f.get("avoid_colors", [])]))
            if f.get("top5_items"):
                st.markdown("- 기본 아이템 Top5:\n" + "\n".join([f"  - {x}" for x in f.get("top5_items", [])]))

        st.markdown("#### 🧍 행동/라이프스타일")
        st.markdown(f"- 제스처/톤: {b.get('gesture_tone','')}")
        st.markdown(f"- 말투/매너: {b.get('speech_manner','')}")
        if b.get("daily_habits"):
            st.markdown("- 작은 습관:\n" + "\n".join([f"  - {x}" for x in b.get("daily_habits", [])]))

        st.divider()
        st.subheader("📷 사용자 사진 업로드(다음 단계)")
        st.caption("현재는 사진 '내용'을 분석하지 않습니다. 대신 체크리스트를 만들어드려요.")
        u2 = st.file_uploader("화장/스타일 사진 업로드(UI만)", type=["jpg", "jpeg", "png"], key="future_photo")
        if u2 is not None:
            st.success("업로드 완료! (현재 단계에서는 이미지 내용은 보지 않아요.)")
            if st.button("✅ 추구미 기준 체크리스트 생성", use_container_width=True):
                if not openai_key:
                    st.warning("OpenAI API Key를 입력해 주세요.")
                else:
                    with st.spinner("체크리스트 생성 중..."):
                        try:
                            sp = (
                                "당신은 추구미 스타일 코치입니다. "
                                "사용자가 목표로 한 추구미 리포트를 바탕으로, "
                                "사용자 사진을 '보지 않는다'는 전제에서 점검 체크리스트를 작성하세요. "
                                "반드시 (1)잘된 점 체크 (2)개선점 체크 (3)대체 방향 제시로 구성. "
                                "JSON이 아니라 일반 텍스트로 간결하게."
                            )
                            uprompt = (
                                "추구미 리포트 요약:\n"
                                f"- 타입: {r.get('type_name_ko','')} / {r.get('type_name_en','')}\n"
                                f"- 한줄 정의: {r.get('identity_one_liner','')}\n"
                                f"- 핵심 키워드: {', '.join(r.get('core_keywords') or [])}\n"
                                f"- 메이크업: {json.dumps(m, ensure_ascii=False)}\n"
                                f"- 패션: {json.dumps(f, ensure_ascii=False)}\n\n"
                                "요청: 사진을 보지 않는 조건에서, 사용자가 스스로 점검할 체크리스트를 만들어줘."
                            )
                            # stream as normal (single placeholder)
                            with st.chat_message("assistant"):
                                txt = openai_stream_chat(
                                    openai_key,
                                    sp,
                                    [{"role": "user", "content": uprompt}],
                                    temperature=0.4,
                                )
                                st.session_state["style_self_checklist"] = txt
                        except Exception as e:
                            st.error(f"체크리스트 오류: {e}")

        if st.session_state.get("style_self_checklist"):
            st.markdown("### 🧾 체크리스트")
            st.markdown(st.session_state["style_self_checklist"])

# -----------------------------
# Controlled tab jump (rerun-based)
# -----------------------------
if st.session_state.get("active_tab", 0) != 0:
    # We can't directly programmatically switch st.tabs reliably,
    # so we use rerun hint + user experience (most Streamlit versions).
    # If user clicked "추구미 설계 시작", we already reran.
    pass
