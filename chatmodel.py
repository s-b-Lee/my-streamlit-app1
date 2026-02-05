import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# Page Config
# =========================
st.set_page_config(page_title="통합 AI 앱: 상담사 → 추구미 설계", page_icon="🧠", layout="wide")

# =========================
# Session State Init
# =========================
def _init_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    if "category" not in st.session_state:
        st.session_state.category = "자기계발"

    if "persona" not in st.session_state:
        st.session_state.persona = "차분한 전문가"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0

    if "last_emotion_label" not in st.session_state:
        st.session_state.last_emotion_label = ""

    if "last_emotion_confidence" not in st.session_state:
        st.session_state.last_emotion_confidence = ""

    if "last_emotion_rationale" not in st.session_state:
        st.session_state.last_emotion_rationale = ""

    if "last_user_text" not in st.session_state:
        st.session_state.last_user_text = ""

    if "suggest_style_bridge" not in st.session_state:
        st.session_state.suggest_style_bridge = False

    if "counsel_summary" not in st.session_state:
        st.session_state.counsel_summary = ""

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "AI 상담사"

    if "mood_logs" not in st.session_state:
        st.session_state.mood_logs = []

    if "style_inputs" not in st.session_state:
        st.session_state.style_inputs = {
            "selected_cards": [],
            "dislikes": "",
            "wants": "",
            "constraints": "",
            "places": [],
            "from_counsel_summary": "",
        }

    if "style_report" not in st.session_state:
        st.session_state.style_report = ""


_init_state()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ 설정")

    st.session_state.api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
        key="openai_key_input",
    )

    st.session_state.category = st.selectbox(
        "상담/코칭 카테고리",
        ["자기계발", "커리어", "연애", "인간관계", "기타"],
        index=["자기계발", "커리어", "연애", "인간관계", "기타"].index(st.session_state.category),
        key="category_select",
    )

    st.session_state.persona = st.selectbox(
        "대화 톤(캐릭터)",
        ["친한 친구", "차분한 전문가", "코치 스타일"],
        index=["친한 친구", "차분한 전문가", "코치 스타일"].index(st.session_state.persona),
        key="persona_select",
    )

    show_notice = st.checkbox("개인정보/의료 고지 보기", value=True, key="notice_checkbox")
    if show_notice:
        st.info(
            "이 앱은 의료/법률 진단을 제공하지 않습니다. "
            "위험하거나 긴급한 상황(자해/자살 등)이 있다면 즉시 주변의 도움을 요청하고 "
            "지역 응급 번호 또는 전문기관에 연락하세요."
        )

    if st.button("🧹 대화 초기화", key="reset_btn"):
        st.session_state.messages = []
        st.session_state.turn_count = 0
        st.session_state.last_emotion_label = ""
        st.session_state.last_emotion_confidence = ""
        st.session_state.last_emotion_rationale = ""
        st.session_state.last_user_text = ""
        st.session_state.suggest_style_bridge = False
        st.session_state.counsel_summary = ""
        st.rerun()

    st.caption("🔒 API 키는 세션에만 유지됩니다(저장되지 않음).")

# =========================
# OpenAI Client
# =========================
def get_client() -> OpenAI:
    return OpenAI(api_key=st.session_state.api_key)

# =========================
# Safety & Heuristics
# =========================
CRISIS_PATTERNS = [
    r"\b자살\b",
    r"\b죽고\s*싶\b",
    r"\b자해\b",
    r"\b해치고\s*싶\b",
    r"\b목숨\b",
    r"\b극단적\s*선택\b",
    r"\b살\s*의미\b",
]

STYLE_BRIDGE_PATTERNS = [
    "이미지", "분위기", "정체성", "추구미", "첫인상", "스타일", "외모", "자신감",
    "옷", "패션", "메이크업", "화장", "인상", "브랜딩", "이미지메이킹",
]

def is_crisis(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(re.search(pat, t) for pat in CRISIS_PATTERNS)

def wants_style_bridge(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(k in t for k in STYLE_BRIDGE_PATTERNS)

# =========================
# Emotion labeling (rule-based)
# =========================
EMOTION_KEYWORDS = {
    "불안": ["불안", "초조", "걱정", "긴장", "두려", "무섭"],
    "슬픔": ["슬프", "우울", "눈물", "허무", "상실", "외롭"],
    "분노": ["화나", "짜증", "분노", "열받", "억울"],
    "지침": ["지쳐", "피곤", "번아웃", "무기력", "힘들", "기진맥진"],
    "설렘": ["설레", "기대", "두근", "좋아", "행복"],
    "부끄러움": ["민망", "부끄", "창피"],
}

def label_emotion(text: str) -> Tuple[str, str, str]:
    t = (text or "").lower()
    scores = {k: 0 for k in EMOTION_KEYWORDS.keys()}
    for label, kws in EMOTION_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[label] += 1

    best = max(scores, key=lambda k: scores[k]) if max(scores.values()) > 0 else "복합/모호"
    confidence = "높음" if best != "복합/모호" and scores[best] >= 2 else ("보통" if best != "복합/모호" else "낮음")
    rationale = "텍스트에서 감정 단서 키워드가 관찰되었습니다." if best != "복합/모호" else "명확한 단서가 부족해 복합 감정으로 추정합니다."
    return best, confidence, rationale

# =========================
# Counseling System Prompts
# =========================
def persona_instructions(persona: str) -> str:
    if persona == "친한 친구":
        return (
            "말투는 친근하고 다정하게. 단, 장황하지 말고 핵심만. "
            "오버 공감/과장 금지. '해요체' 유지."
        )
    if persona == "코치 스타일":
        return (
            "말투는 코치처럼 단호하지만 따뜻하게. "
            "문제 정의→선택지→다음 행동 2~3개로 구조화."
        )
    return (
        "말투는 차분한 전문가(교수님 느낌). "
        "두괄식, 논리적, 필요한 말만. 비유는 최대 1회."
    )

def category_frame(category: str) -> str:
    frames = {
        "자기계발": "프레임: 현재 상태→원하는 변화→방해요인→작은 행동(오늘/이번주)→피드백 루프.",
        "커리어": "프레임: 목표→강점/갭→우선순위(1~2개)→실행 계획(작업 단위)→리스크 대비.",
        "연애": "프레임: 관계 목표→상황 분석→내 감정/욕구→경계/소통 문장→다음 행동.",
        "인간관계": "프레임: 갈등 원인→내 역할→상대 관점→대화 스크립트→후속 행동.",
        "기타": "프레임: 문제 정의→원인 가설→선택지→다음 행동.",
    }
    return frames.get(category, frames["기타"])

COUNSEL_SYSTEM = """
당신은 대학생/대학원생을 돕는 대화형 AI 상담/코칭 비서입니다.

핵심 패턴(항상 적용):
- 1) 즉시 공감(짧게) → 2) 문제를 한 문장으로 정리 → 3) 구체적 행동 제안 2~3개(오늘/이번주 단위) → 4) 확인 질문 1개
- 두괄식, 장황하지 않게, 논리적.
- 이해를 돕는 비유는 최대 1회만 사용.
- 없는 사실 지어내지 말고, 불확실하면 '추정'으로 표시.
- 상담은 의료/진단이 아니다. 의학적 판단은 하지 않는다.

위험 신호(자해/자살/타해 등)가 보이면:
- 즉시 안전 안내/전문기관 권유 문구를 출력
- 추가로 안전 확인 질문 1개만 한다
- 그 외의 코칭은 진행하지 않는다

정기 요약:
- 사용자 턴 기준 6~8턴마다:
  '요약(3줄 이내) + 다음 행동 2~3개'를 반드시 포함한다.

추구미로의 자연스러운 연결:
- 사용자의 고민이 이미지/분위기/정체성/첫인상/스타일/외모/자신감과 연관되어 보이면,
  대답 말미에 한 문장으로 "추구미 설계로 이어가도 될까요?"를 제안한다.
""".strip()

STYLE_SYSTEM = """
당신은 '추구미(이미지 정체성) 설계' AI입니다.
목표는 사용자가 어떤 사람처럼 보이고 싶은지(분위기/정체성)를 구조화된 언어로 정리하고,
화장/패션/태도/라이프스타일까지 실행 가이드를 제시하는 것입니다.

규칙:
- 브랜드/제품 추천 금지(방향성 중심).
- '좋다/나쁘다' 판단 금지. 추구미 기준에서의 적합성으로 피드백.
- 출력은 아래 섹션을 지켜라.

출력 형식:
1) 핵심 키워드 3~5개
2) 추구미 타입명(가능하면 국문 + 영문)
3) 한 문장 정체성 정의
4) 미니 리포트: 분위기 요약 / 타인에게 주는 인상 / 잘 어울리는 상황 / 과도함 주의 포인트 / 유지 난이도(낮/중/높)
5) 실천 가이드:
   - 메이크업 방향성: 베이스 / 포인트(눈·입) / 피하면 좋은 요소
   - 패션 방향성: 실루엣 / 컬러 팔레트 / 피하면 좋은 컬러 / 기본 아이템 우선순위 Top5
6) (있다면) 상담 요약 반영 전략: 현재 고민을 고려해 실천 난이도를 낮추는 방식으로 조정안을 제시
""".strip()

# =========================
# Streaming (single placeholder)
# =========================
def stream_chat_completion(client: OpenAI, messages: List[Dict[str, str]], system_prompt: str) -> str:
    placeholder = st.empty()
    acc = ""

    stream = client.responses.create(
        model="gpt-4-mini",
        instructions=system_prompt,
        input=messages,
        stream=True,
    )

    for event in stream:
        etype = getattr(event, "type", None)
        if etype is None and isinstance(event, dict):
            etype = event.get("type")

        if etype == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if delta is None and isinstance(event, dict):
                delta = event.get("delta", "")
            if delta:
                acc += delta
                placeholder.markdown(acc)

        if etype in ("response.completed", "response.done"):
            break

    placeholder.markdown(acc)
    return acc

# =========================
# Counseling summary helper
# =========================
def build_counsel_summary() -> str:
    msgs = st.session_state.messages[-10:]
    user_texts = [m["content"] for m in msgs if m["role"] == "user"]
    assistant_texts = [m["content"] for m in msgs if m["role"] == "assistant"]

    def _clip(s: str, n: int = 180) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[:n].rstrip() + "…"

    core_user = _clip(" / ".join(user_texts[-3:]), 240) if user_texts else ""
    core_assistant = _clip(" / ".join(assistant_texts[-2:]), 240) if assistant_texts else ""

    pieces = []
    if core_user:
        pieces.append(f"- 최근 상황(사용자): {core_user}")
    if st.session_state.last_emotion_label:
        pieces.append(f"- 추정 감정: {st.session_state.last_emotion_label} (신뢰도: {st.session_state.last_emotion_confidence})")
    if core_assistant:
        pieces.append(f"- 최근 조언 요약: {core_assistant}")
    return "\n".join(pieces).strip()

# =========================
# Mood tracker helpers (NO matplotlib)
# =========================
MOOD_CHOICES = [
    ("😊", "좋음"),
    ("😌", "평온"),
    ("😐", "무덤덤"),
    ("😟", "불안"),
    ("😞", "우울"),
    ("😠", "화남"),
    ("😫", "지침"),
]

MOOD_SCORE = {
    "좋음": 5,
    "평온": 4,
    "무덤덤": 3,
    "불안": 2,
    "우울": 1,
    "화남": 1,
    "지침": 1,
}

TRIGGER_WORDS = ["시험", "과제", "회의", "면접", "데이트", "발표", "교수", "팀플", "취업", "연애", "친구", "가족", "동아리"]

def add_mood_log(mood_label: str, emoji: str, memo: str, ai_label: str):
    now = datetime.now()
    st.session_state.mood_logs.append(
        {
            "ts": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "weekday": now.strftime("%a"),
            "hour": now.hour,
            "mood": mood_label,
            "emoji": emoji,
            "memo": (memo or "").strip(),
            "ai_label": (ai_label or "").strip(),
        }
    )

def mood_df() -> pd.DataFrame:
    if not st.session_state.mood_logs:
        return pd.DataFrame(columns=["date", "weekday", "hour", "emoji", "mood", "memo", "ai_label", "ts"])
    return pd.DataFrame(st.session_state.mood_logs)

def insight_from_logs(df: pd.DataFrame) -> str:
    if df.empty:
        return "기록이 쌓이면 '자주 등장하는 감정/트리거' 인사이트를 보여줄게요."
    all_text = " ".join((df["memo"].fillna("") + " " + df["ai_label"].fillna("")).tolist())
    hits = [w for w in TRIGGER_WORDS if w in all_text]
    top_moods = df["mood"].value_counts().head(3).to_dict()
    mood_part = ", ".join([f"{k}({v})" for k, v in top_moods.items()]) if top_moods else "없음"
    trigger_part = ", ".join(hits[:8]) if hits else "뚜렷한 트리거 키워드가 아직은 적어요."

    caution = (
        "반복적으로 우울/불안/지침이 지속되거나 일상 기능이 떨어진다면, "
        "스트레스/수면/호르몬(PMS 포함) 등 다양한 요인이 있을 수 있어요. "
        "가능하면 전문가 상담/진료를 고려해보는 것도 방법입니다(진단은 여기서 할 수 없어요)."
    )
    return f"자주 기록된 기분: {mood_part}\n\n자주 등장한 상황 키워드: {trigger_part}\n\n{caution}"

# =========================
# Header
# =========================
st.title("🧠 통합 AI: 상담사 → 감정 트래커 → 추구미 설계")
st.caption("대학생/대학원생을 위한 멘탈케어·코칭과 이미지 정체성(추구미) 설계를 한 곳에서.")

TAB_NAMES = ["AI 상담사", "감정 트래커", "추구미 설계"]
tab_choice = st.radio("탭 선택", TAB_NAMES, horizontal=True, index=TAB_NAMES.index(st.session_state.active_tab))
st.session_state.active_tab = tab_choice
st.divider()

# =========================
# TAB 1: AI 상담사
# =========================
if st.session_state.active_tab == "AI 상담사":
    if not st.session_state.api_key.strip():
        st.warning("사이드바에 OpenAI API Key를 입력하면 상담이 활성화됩니다.")
        st.stop()

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Emotion bar + save
    if st.session_state.last_emotion_label:
        st.info(
            f"🧩 감정 라벨(추정): **{st.session_state.last_emotion_label}** "
            f"(신뢰도: {st.session_state.last_emotion_confidence})"
        )
        cols = st.columns([1, 2, 2])
        with cols[0]:
            if st.button("📝 오늘 감정으로 저장", key="save_emotion_btn"):
                add_mood_log(
                    mood_label="무덤덤",
                    emoji="📝",
                    memo=st.session_state.last_user_text,
                    ai_label=st.session_state.last_emotion_label,
                )
                st.success("감정 트래커에 저장했어요.")
        with cols[1]:
            st.caption("감정 트래커 탭에서 기분(이모지)을 더 정확히 선택해 저장할 수 있어요.")
        with cols[2]:
            st.caption("※ 이 라벨은 추정이며, 진단이 아닙니다.")

    # Style bridge CTA
    if st.session_state.suggest_style_bridge:
        st.success("✨ 추구미(이미지 정체성) 설계로 이어가면 도움이 될 것 같아요.")
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🎨 추구미 설계 시작", key="go_style_btn"):
                st.session_state.counsel_summary = build_counsel_summary()
                st.session_state.style_inputs["from_counsel_summary"] = st.session_state.counsel_summary
                st.session_state.active_tab = "추구미 설계"
                st.rerun()
        with c2:
            st.caption("상담 내용을 3줄로 요약해 추구미 탭에 자동으로 넘겨요.")

    user_text = st.chat_input("지금 고민/상황을 적어주세요. (예: 요즘 지치고 불안해요)", key="counsel_input")
    if user_text:
        st.session_state.turn_count += 1
        st.session_state.last_user_text = user_text

        # Crisis
        if is_crisis(user_text):
            with st.chat_message("assistant"):
                st.markdown(
                    "지금은 **안전이 최우선**이에요.\n\n"
                    "- 주변에 믿을 수 있는 사람에게 **지금 바로** 도움을 요청해 주세요.\n"
                    "- 긴급한 위험이 느껴지면 **지역 응급 번호**(예: 112/119 등) 또는 가까운 응급실로 연락/이동을 권합니다.\n"
                    "- 자해/자살 생각이 강하거나 계획이 있다면, 혼자 있지 말고 즉시 도움을 받는 게 필요합니다.\n\n"
                    "**지금 혼자 계신가요, 아니면 옆에 누군가 있나요?**"
                )
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append(
                {"role": "assistant", "content": "안전이 최우선입니다. 지금 혼자 계신가요, 아니면 옆에 누군가 있나요?"}
            )
            st.stop()

        emo, conf, rat = label_emotion(user_text)
        st.session_state.last_emotion_label = emo
        st.session_state.last_emotion_confidence = conf
        st.session_state.last_emotion_rationale = rat

        if wants_style_bridge(user_text):
            st.session_state.suggest_style_bridge = True

        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        system_prompt = (
            COUNSEL_SYSTEM
            + "\n\n"
            + f"사용자 카테고리: {st.session_state.category}\n"
            + f"{category_frame(st.session_state.category)}\n"
            + f"말투 지침: {persona_instructions(st.session_state.persona)}\n"
            + f"추정 감정 라벨(참고): {emo} (신뢰도: {conf})\n"
            + "주의: 의학적 진단 금지. 불확실하면 추정이라고 말할 것.\n"
        ).strip()

        if st.session_state.turn_count % 7 == 0:
            system_prompt += "\n\n이번 답변에는 반드시 '요약(3줄) + 다음 행동 2~3개'를 포함해라."

        client = get_client()
        with st.chat_message("assistant"):
            try:
                assistant_text = stream_chat_completion(client, st.session_state.messages, system_prompt)
            except Exception as e:
                st.error(
                    "⚠️ OpenAI API 호출 오류\n\n"
                    f"{e}\n\n"
                    "체크리스트:\n"
                    "- API 키가 올바른지\n"
                    "- 네트워크/방화벽\n"
                    "- 사용량/레이트리밋\n"
                )
                st.stop()

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        if "추구미" in assistant_text or "이미지" in assistant_text:
            st.session_state.suggest_style_bridge = True

        st.rerun()

# =========================
# TAB 2: 감정 트래커
# =========================
elif st.session_state.active_tab == "감정 트래커":
    st.subheader("📌 감정 트래커")
    st.write("오늘의 기분을 기록하고, 패턴을 간단히 분석해요.")

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        mood_pick = st.selectbox(
            "오늘 기분",
            [f"{e} {m}" for e, m in MOOD_CHOICES],
            index=2,
            key="mood_pick",
        )
        emoji = mood_pick.split(" ")[0]
        mood_label = " ".join(mood_pick.split(" ")[1:])

    with c2:
        memo = st.text_input("간단 메모", placeholder="예: 팀플 회의 후 기분이 가라앉았음", key="mood_memo")

    with c3:
        ai_label_hint = st.text_input(
            "AI 감정 라벨(선택)",
            value=st.session_state.last_emotion_label,
            placeholder="예: 불안/지침/분노…",
            key="mood_ai_label",
        )

    if st.button("💾 저장", key="mood_save_btn"):
        add_mood_log(mood_label=mood_label, emoji=emoji, memo=memo, ai_label=ai_label_hint)
        st.success("저장했어요!")
        st.rerun()

    st.divider()
    df = mood_df()

    st.subheader("🗂️ 기록")
    st.dataframe(df[["date", "weekday", "hour", "emoji", "mood", "ai_label", "memo"]], use_container_width=True)

    st.divider()
    st.subheader("📈 패턴 분석")
    if df.empty:
        st.info("아직 기록이 없어요.")
    else:
        tmp = df.copy()
        tmp["score"] = tmp["mood"].map(MOOD_SCORE).fillna(3)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**요일별 기분 변화(평균)**")
            order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            g = tmp.groupby("weekday", as_index=False)["score"].mean()
            g["weekday"] = pd.Categorical(g["weekday"], categories=order, ordered=True)
            g = g.sort_values("weekday")
            st.line_chart(g.set_index("weekday")["score"])

        with cB:
            st.markdown("**시간대별 기분 변화(평균)**")
            h = tmp.groupby("hour", as_index=False)["score"].mean().sort_values("hour")
            st.line_chart(h.set_index("hour")["score"])

    st.divider()
    st.subheader("💡 인사이트")
    st.write(insight_from_logs(df))

# =========================
# TAB 3: 추구미 설계
# =========================
else:
    if not st.session_state.api_key.strip():
        st.warning("사이드바에 OpenAI API Key를 입력하면 추구미 분석이 활성화됩니다.")
        st.stop()

    st.subheader("🎨 추구미(이미지 정체성) 설계")
    st.write("‘예뻐지는 법’이 아니라, **내가 어떤 사람처럼 보이고 싶은지**를 구조화하고 실행으로 연결해요.")

    if st.session_state.style_inputs.get("from_counsel_summary"):
        with st.expander("🧾 상담 요약(자동 전달됨)", expanded=True):
            st.markdown(st.session_state.style_inputs["from_counsel_summary"])

    st.divider()

    st.markdown("### 1) 끌리는 키워드 카드를 5~10개 선택하세요")
    mood_cards = ["청순", "시크", "힙", "차분", "관능", "내추럴"]
    style_cards = ["미니멀", "스트릿", "클래식", "Y2K", "캐주얼", "포멀"]

    selected = st.multiselect(
        "무드/스타일 카드",
        options=mood_cards + style_cards,
        default=st.session_state.style_inputs.get("selected_cards", []),
        key="style_cards_multiselect",
        help="로컬 이미지 없이도 테스트 가능한 임시 카드 UI입니다.",
    )

    st.markdown("### 2) 텍스트로 보조 입력")
    dislikes = st.text_area(
        "이런 느낌은 싫어요",
        value=st.session_state.style_inputs.get("dislikes", ""),
        placeholder="예: 너무 꾸민 느낌, 과한 펄, 튀는 색감",
        key="style_dislikes",
        height=70,
    )
    wants = st.text_area(
        "원하는 느낌(또는 원하는 변화)",
        value=st.session_state.style_inputs.get("wants", ""),
        placeholder="예: 편해 보이는데 세련됐으면, 신뢰감 있어 보이고 싶어요",
        key="style_wants",
        height=70,
    )
    constraints = st.text_area(
        "제약/조건(선택)",
        value=st.session_state.style_inputs.get("constraints", ""),
        placeholder="예: 예산, 교복/정장, 피부 타입, 실습/알바 환경 등",
        key="style_constraints",
        height=70,
    )

    st.markdown("### 3) 이 추구미로 주로 가고 싶은 공간(복수 선택)")
    places = st.multiselect(
        "공간 선택",
        options=["학교", "직장", "데이트", "SNS", "공식 자리"],
        default=st.session_state.style_inputs.get("places", []),
        key="style_places",
    )

    st.divider()

    st.markdown("### (선택) 내 사진 업로드")
    st.caption("현재 버전에서는 이미지 내용 분석은 하지 않아요. 대신 ‘추구미 기준 체크리스트’만 생성해요.")
    uploaded = st.file_uploader("화장/스타일 사진 업로드", type=["png", "jpg", "jpeg"], key="style_photo")

    st.session_state.style_inputs.update(
        {
            "selected_cards": selected,
            "dislikes": dislikes,
            "wants": wants,
            "constraints": constraints,
            "places": places,
        }
    )

    def style_payload() -> Dict[str, Any]:
        return {
            "selected_cards": selected,
            "dislikes": dislikes,
            "wants": wants,
            "constraints": constraints,
            "places": places,
            "counsel_summary": st.session_state.style_inputs.get("from_counsel_summary", ""),
            "note": "브랜드/제품 추천 금지. 방향성 중심. 진단/단정 금지.",
        }

    analyze_btn = st.button("✨ 추구미 분석", key="style_analyze_btn", use_container_width=True)

    if analyze_btn:
        if len(selected) < 5:
            st.warning("카드를 5개 이상 선택해 주세요.")
            st.stop()

        client = get_client()
        payload = style_payload()

        user_msgs = [
            {
                "role": "user",
                "content": (
                    "아래 입력을 바탕으로 추구미 리포트와 실천 가이드를 작성해줘.\n"
                    f"입력(JSON): {json.dumps(payload, ensure_ascii=False)}"
                ),
            }
        ]

        with st.spinner("분석 중..."):
            try:
                report = stream_chat_completion(client, user_msgs, STYLE_SYSTEM)
            except Exception as e:
                st.error(
                    "⚠️ OpenAI API 호출 오류\n\n"
                    f"{e}\n\n"
                    "체크리스트:\n"
                    "- API 키가 올바른지\n"
                    "- 네트워크/방화벽\n"
                    "- 사용량/레이트리밋\n"
                )
                st.stop()

        st.session_state.style_report = report
        st.rerun()

    if st.session_state.style_report:
        st.divider()
        st.markdown("## 📄 나의 추구미 리포트")
        st.markdown(st.session_state.style_report)

    if uploaded is not None:
        st.divider()
        st.markdown("## ✅ 추구미 기준 체크리스트(사진 내용은 분석하지 않음)")

        checklist_prompt = {
            "selected_cards": selected,
            "wants": wants,
            "dislikes": dislikes,
            "places": places,
            "instruction": (
                "사용자가 사진을 업로드했지만, 이미지 내용은 보지 않는다고 명시하고, "
                "추구미 기준으로 스스로 점검할 체크리스트를 생성해줘. "
                "구성: 잘된 점(자기점검 항목) / 개선 제안(점검 항목) / 대체 방향(선택지). "
                "브랜드/제품 추천 금지."
            ),
        }

        client = get_client()
        with st.spinner("체크리스트 생성 중..."):
            try:
                checklist = stream_chat_completion(
                    client,
                    [{"role": "user", "content": json.dumps(checklist_prompt, ensure_ascii=False)}],
                    "당신은 추구미 코치입니다. 체크리스트만 간결하게 작성하세요.",
                )
            except Exception as e:
                st.error(f"오류: {e}")
                st.stop()

        st.markdown(checklist)
