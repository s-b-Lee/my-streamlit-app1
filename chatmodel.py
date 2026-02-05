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
    st.session_state.setdefault("followup_question", "")

init_state()

# =========================
# Helpers
# =========================
def get_client() -> OpenAI:
    return OpenAI(api_key=st.session_state.openai_api_key)

def b64_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    # Responses API 이미지 입력은 image_url에 data URL 형태 지원 (docs 예시: data:image/jpeg;base64,...)
    return f"data:{mime};base64,{b64}"

def safe_trim_images(files: List, max_images: int = 3) -> List:
    if not files:
        return []
    return files[:max_images]

def stream_response_text(
    client: OpenAI,
    system_instructions: str,
    input_items: List[Dict],
) -> str:
    """
    Stream text with ONE placeholder (stable rendering).
    Uses Responses API streaming events: response.output_text.delta
    """
    ph = st.empty()
    acc = ""

    stream = client.responses.create(
        model=st.session_state.model,
        instructions=system_instructions,
        input=input_items,
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
                ph.markdown(acc)

        if etype in ("response.completed", "response.done"):
            break

    ph.markdown(acc)
    return acc

def now_kst_str() -> str:
    # Streamlit Cloud/로컬 타임존이 다를 수 있어도 일단 ISO로 기록
    return datetime.now().isoformat(timespec="seconds")

# =========================
# System Prompts
# =========================
STYLE_SYSTEM = """
당신은 '추구미(이미지 정체성) 설계' 전문 AI입니다.

서비스 정의:
- 사용자가 어떤 사람처럼 보이고 싶은지(이미지/분위기/정체성)를 구조화된 언어로 정의하고,
  화장/패션/태도/라이프스타일까지 연결해 실천하도록 돕는다.
- 단순 미용/패션 추천이 아니라 '정체성 설계 도구'다.

절대 규칙:
- 브랜드/제품 추천 금지(방향성 중심).
- 단정/비하/외모 평가 금지. "추구미 기준 적합성"으로만 말한다.
- 의료/심리 진단 금지.
- 결과는 두괄식, 불필요한 말 없이.

출력 형식(반드시 유지):
1) 핵심 키워드 3~5개
2) 추구미 타입명 (국문 + 영문)
3) 한 문장 정체성 정의
4) 미니 리포트
   - 전체 분위기 요약
   - 타인에게 주는 인상
   - 잘 어울리는 상황(학교/직장/데이트/SNS/공식 자리 등과 연결)
   - 과도함 주의 포인트
   - 유지 난이도(낮/중/높) + 이유 1줄
5) 실천 가이드(방향성만)
   - 메이크업: 베이스 / 눈 / 입 / 피하면 좋은 요소
   - 패션: 실루엣 / 컬러 팔레트 / 피하면 좋은 컬러 / 기본 아이템 Top5
6) 다음 실험(1주 플랜)
   - 이번 주에 바로 해볼 수 있는 작은 실험 3개
""".strip()

INSPO_IMAGE_SYSTEM = """
당신은 '추구미 레퍼런스(인스포) 이미지 분석가'다.
사용자가 업로드한 '좋다고 느꼈던 이미지'들을 보고(이미지 자체를 분석),
공통된 무드/스타일 신호를 추출해 '추구미 설계'에 쓸 수 있는 구조화된 요약을 만든다.

규칙:
- 사람 얼굴/체형/외모 평가 금지.
- 브랜드/제품 추정 금지.
- 이미지 속에서 관찰되는 요소(실루엣, 소재 느낌, 색감, 대비, 광택, 디테일, 무드, TPO)를 중심으로 말한다.
- 추측이 필요하면 "추정"으로 표시.

출력 형식(반드시 유지):
A) 공통 무드 키워드 5개
B) 스타일 신호 6개(예: 직선/곡선, 미니멀/디테일, 대비, 소재 질감, 컬러 톤, 액세서리 밀도 등)
C) 추천 컬러 톤 4개(예: 뉴트럴/저채도/고채도 포인트 등) + 피할 톤 2개
D) 추구미 문장 1개(한 줄)
E) 다음 단계 질문 3개(사용자에게 확인할 질문)
""".strip()

FIT_FEEDBACK_SYSTEM = """
당신은 '추구미 적합성 피드백' 전문 AI다.
사용자가 업로드한 본인 스타일 사진(또는 현재 스타일을 보여주는 이미지)을 보고,
이미 정의된 추구미(키워드/타입/문장)를 기준으로 "적합성"을 피드백한다.

규칙:
- 외모 비하/판단 금지. 아름다움 평가 금지.
- 체형/민감 특성 추정 금지.
- 브랜드/제품 추천 금지(방향성 제시만).
- 수치(%)는 정밀 측정이 아니라 "체감 적합성"으로 제시하고, 근거는 관찰 기반으로 짧게.

출력 형식(반드시 유지):
1) 추구미 일치도: XX%
2) 잘된 점 3개 (추구미 기준)
3) 어긋난 신호 3개 (추구미 기준)
4) 개선 제안 3개 (방향성)
5) 대체 방향 2개 (선택지)
""".strip()

# =========================
# UI: Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ 설정")

    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-...",
    )

    st.session_state.model = st.selectbox(
        "모델",
        options=["gpt-4-mini"],
        index=0,
        help="요구사항에 맞춰 gpt-4-mini 사용",
    )

    st.divider()
    st.caption("🔒 키는 세션에만 유지됩니다(저장되지 않음).")

    if st.button("🧹 전체 초기화"):
        for k in [
            "selected_cards", "dislikes", "wants", "constraints", "places", "notes",
            "style_report", "inspo_analysis", "fit_feedback",
            "style_logs", "last_profile_summary", "followup_question",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        init_state()
        st.rerun()

# =========================
# Main Header
# =========================
st.title("🎨 추구미(이미지 정체성) 설계 AI")
st.caption("‘예뻐지는 법’이 아니라, **내가 어떤 사람처럼 보이고 싶은지**를 구조화하고 실행으로 연결합니다.")

if not st.session_state.openai_api_key.strip():
    st.warning("사이드바에 OpenAI API Key를 입력하면 기능이 활성화됩니다.")
    st.stop()

tabs = st.tabs(["① 추구미 발견", "② 추구미 리포트", "③ 사진 기반 피드백", "④ 추구미 트래커"])

# =========================
# TAB 1: Discovery (incl. 2-3 image upload)
# =========================
with tabs[0]:
    st.subheader("① 추구미 발견")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 2-1) 무드/스타일 선택(카드 대체 UI)")
        mood_cards = ["청순", "시크", "힙", "차분", "관능", "내추럴"]
        style_cards = ["미니멀", "스트릿", "클래식", "Y2K", "캐주얼"]
        st.session_state.selected_cards = st.multiselect(
            "끌리는 키워드(5~10개 추천)",
            options=mood_cards + style_cards,
            default=st.session_state.selected_cards,
        )

    with c2:
        st.markdown("#### 2-2) 텍스트 보조 입력")
        st.session_state.dislikes = st.text_area(
            "이런 느낌은 싫어요",
            value=st.session_state.dislikes,
            placeholder="예: 너무 꾸민 느낌 말고, 과한 펄/고채도, 과한 로고, 답답한 인상",
            height=80,
        )
        st.session_state.wants = st.text_area(
            "원하는 느낌/한 문장",
            value=st.session_state.wants,
            placeholder="예: 편해 보이는데 세련됐으면, 신뢰감+차분함, 가까이 가기 쉬운 단정함",
            height=80,
        )
        st.session_state.constraints = st.text_area(
            "제약/조건(선택)",
            value=st.session_state.constraints,
            placeholder="예: 학교/실습 환경, 예산, 피부 표현 선호(매트/세미글로우), 활동량",
            height=70,
        )

    st.markdown("#### 2-4) 상황 기반 질문(공간 선택)")
    st.session_state.places = st.multiselect(
        "이 추구미로 주로 가고 싶은 공간",
        options=["학교", "직장", "데이트", "SNS", "공식 자리"],
        default=st.session_state.places,
    )

    st.divider()

    st.markdown("#### 2-3) 이미지 업로드(핵심) — 내가 ‘좋다’고 느낀 레퍼런스")
    st.caption("여기 업로드한 이미지의 **무드/스타일 신호를 실제로 분석**해서 추구미 설계에 반영합니다. (최대 3장 권장)")
    inspo_files = st.file_uploader(
        "레퍼런스 이미지 업로드(선택)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="inspo_files",
    )

    analyze_inspo = st.button("🔍 레퍼런스 이미지에서 추구미 신호 뽑기", use_container_width=True)

    if analyze_inspo:
        files = safe_trim_images(inspo_files or [], max_images=3)
        if not files:
            st.warning("레퍼런스 이미지를 1장 이상 업로드해 주세요.")
            st.stop()

        client = get_client()

        content_parts = [{"type": "input_text", "text": "업로드된 레퍼런스 이미지들을 분석해 추구미 신호를 구조화해줘."}]
        for f in files:
            mime = f.type or "image/jpeg"
            data_url = b64_data_url(f.getvalue(), mime)
            content_parts.append({"type": "input_image", "image_url": data_url})

        input_items = [{"role": "user", "content": content_parts}]

        with st.spinner("이미지 분석 중..."):
            try:
                text = stream_response_text(client, INSPO_IMAGE_SYSTEM, input_items)
            except Exception as e:
                st.error(f"OpenAI API 오류: {e}")
                st.stop()

        st.session_state.inspo_analysis = text
        st.rerun()

    if st.session_state.inspo_analysis:
        st.markdown("### ✅ 레퍼런스 이미지 기반 요약")
        st.markdown(st.session_state.inspo_analysis)

    st.divider()
    st.session_state.notes = st.text_input(
        "추가로 알려주고 싶은 것(선택)",
        value=st.session_state.notes,
        placeholder="예: 너무 차가워 보이진 않았으면, 하지만 프로페셔널함은 유지하고 싶어요",
    )

# =========================
# TAB 2: Report
# =========================
with tabs[1]:
    st.subheader("② AI 추구미 분석 & 리포트")

    st.caption("선택 카드 + 텍스트 + (있다면) 레퍼런스 이미지 분석 결과를 합쳐 최종 추구미 리포트를 생성합니다.")

    payload = {
        "selected_cards": st.session_state.selected_cards,
        "dislikes": st.session_state.dislikes,
        "wants": st.session_state.wants,
        "constraints": st.session_state.constraints,
        "places": st.session_state.places,
        "inspo_image_analysis": st.session_state.inspo_analysis,
        "notes": st.session_state.notes,
    }

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("#### 입력 요약")
        st.json(payload, expanded=False)
    with colB:
        st.markdown("#### 생성 버튼")
        build_report = st.button("✨ 추구미 리포트 생성", use_container_width=True)

    if build_report:
        if len(st.session_state.selected_cards) < 5 and not st.session_state.inspo_analysis.strip():
            st.warning("키워드를 5개 이상 선택하거나, 레퍼런스 이미지 분석을 먼저 진행해 주세요.")
            st.stop()

        client = get_client()
        input_items = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "아래 입력을 바탕으로 추구미 리포트를 작성해줘."},
                {"type": "input_text", "text": f"입력(JSON): {json.dumps(payload, ensure_ascii=False)}"},
            ],
        }]

        with st.spinner("리포트 생성 중..."):
            try:
                report = stream_response_text(client, STYLE_SYSTEM, input_items)
            except Exception as e:
                st.error(f"OpenAI API 오류: {e}")
                st.stop()

        st.session_state.style_report = report

        # 짧은 프로필 요약(후속 질문/트래커에 활용)
        st.session_state.last_profile_summary = (
            f"키워드: {', '.join(st.session_state.selected_cards[:8])}\n"
            f"원하는 느낌: {st.session_state.wants.strip() or '(미입력)'}\n"
            f"피하고 싶은 것: {st.session_state.dislikes.strip() or '(미입력)'}"
        )
        st.session_state.followup_question = "요즘 환경(학교/직장/대인관계)이 바뀌었나요? 바뀐 점 1~2가지만 적어줘요."
        st.rerun()

    if st.session_state.style_report:
        st.markdown("### 📄 나의 추구미 리포트")
        st.markdown(st.session_state.style_report)

        st.divider()
        st.markdown("### 🔁 팔로업(대화 기억 & 조정)")
        st.caption("추구미는 환경에 따라 조정될 수 있어요. 변화가 있으면 업데이트 제안을 만들어요.")
        user_update = st.text_input("환경 변화/상황 변화(선택)", placeholder=st.session_state.followup_question)
        if st.button("🧩 변화 반영해 조정안 만들기"):
            if not user_update.strip():
                st.warning("변화 내용을 한 줄이라도 적어줘요.")
            else:
                client = get_client()
                input_items = [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "아래 추구미 요약과 변화 내용을 반영해 '조정안'만 간결하게 만들어줘."},
                        {"type": "input_text", "text": f"추구미 요약:\n{st.session_state.last_profile_summary}"},
                        {"type": "input_text", "text": f"변화 내용:\n{user_update.strip()}"},
                    ],
                }]
                with st.spinner("조정안 생성 중..."):
                    try:
                        tweak = stream_response_text(
                            client,
                            "당신은 추구미 설계 AI입니다. 출력: 1) 유지할 것 2) 바꿀 것 3) 이번 주 실험 2개. 브랜드/제품 금지.",
                            input_items,
                        )
                    except Exception as e:
                        st.error(f"OpenAI API 오류: {e}")
                        st.stop()
                st.markdown("#### ✅ 조정안")
                st.markdown(tweak)

# =========================
# TAB 3: Photo-based fit feedback (5. 사용자 스타일 피드백)
# =========================
with tabs[2]:
    st.subheader("③ 사용자 스타일 피드백(사진 기반)")
    st.caption("본인 사진을 업로드하면, **현재 추구미 기준으로** 일치도와 개선점을 제공합니다.")

    if not st.session_state.style_report.strip() and not st.session_state.last_profile_summary.strip():
        st.info("먼저 ② 탭에서 추구미 리포트를 생성하면 더 정확하게 피드백할 수 있어요.")
    else:
        st.success("추구미 기준이 준비됐어요. 사진을 올려 주세요.")

    my_photo = st.file_uploader(
        "내 스타일 사진 업로드(상반신/전신/메이크업 등)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="my_photo",
    )

    run_fit = st.button("📌 내 사진을 추구미 기준으로 피드백", use_container_width=True)

    if run_fit:
        if my_photo is None:
            st.warning("사진을 1장 업로드해 주세요.")
            st.stop()

        client = get_client()
        mime = my_photo.type or "image/jpeg"
        data_url = b64_data_url(my_photo.getvalue(), mime)

        # 추구미 기준 텍스트(리포트 전체를 넣으면 길 수 있어 요약을 우선 사용)
        basis = st.session_state.last_profile_summary.strip() or st.session_state.style_report[:800]

        content_parts = [
            {"type": "input_text", "text": "아래 '추구미 기준'과 업로드된 '내 사진'을 바탕으로 적합성 피드백을 작성해줘."},
            {"type": "input_text", "text": f"추구미 기준(요약):\n{basis}"},
            {"type": "input_image", "image_url": data_url},
        ]
        input_items = [{"role": "user", "content": content_parts}]

        with st.spinner("사진 기반 피드백 생성 중..."):
            try:
                fb = stream_response_text(client, FIT_FEEDBACK_SYSTEM, input_items)
            except Exception as e:
                st.error(f"OpenAI API 오류: {e}")
                st.stop()

        st.session_state.fit_feedback = fb
        st.rerun()

    if st.session_state.fit_feedback:
        st.markdown("### ✅ 추구미 적합성 피드백")
        st.markdown(st.session_state.fit_feedback)

# =========================
# TAB 4: Tracker (6. 유지 & 성장 관리)
# =========================
with tabs[3]:
    st.subheader("④ 추구미 트래커(유지 & 성장 관리)")
    st.caption("오늘의 스타일이 추구미와 얼마나 맞았는지 기록하고, 패턴을 봅니다.")

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        fit_choice = st.selectbox("오늘의 스타일", ["잘 맞음", "애매", "어긋남"], index=1)
    with col2:
        situation = st.text_input("상황/공간(선택)", placeholder="예: 학교 수업 / 데이트 / 발표 / 실습 / 면접")
    with col3:
        memo = st.text_input("짧은 메모(선택)", placeholder="예: 단정했지만 너무 딱딱해 보였다는 피드백")

    if st.button("📝 기록 저장", use_container_width=True):
        st.session_state.style_logs.append(
            {
                "ts": now_kst_str(),
                "fit": fit_choice,
                "situation": situation.strip(),
                "memo": memo.strip(),
            }
        )
        st.success("저장했어요!")
        st.rerun()

    df = pd.DataFrame(st.session_state.style_logs) if st.session_state.style_logs else pd.DataFrame(
        columns=["ts", "fit", "situation", "memo"]
    )

    st.divider()
    st.markdown("### 🗂️ 기록")
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.markdown("### 📈 패턴")
    if df.empty:
        st.info("기록이 쌓이면 요일/상황별 패턴을 보여줄게요.")
    else:
        # 간단 점수화
        score_map = {"잘 맞음": 3, "애매": 2, "어긋남": 1}
        tmp = df.copy()
        tmp["score"] = tmp["fit"].map(score_map).fillna(2)

        # 날짜 파생(로컬 시간 기준)
        tmp["date"] = pd.to_datetime(tmp["ts"]).dt.date.astype(str)

        st.markdown("**날짜별 추구미 적합 점수(평균)**")
        agg = tmp.groupby("date", as_index=False)["score"].mean().sort_values("date")
        st.line_chart(agg.set_index("date")["score"])

        st.markdown("**상황 키워드(간단)**")
        # 상황 텍스트에서 상위 키워드 간단 추출(룰 기반)
        text = " ".join(tmp["situation"].fillna("").tolist())
        candidates = [w for w in ["학교", "직장", "데이트", "SNS", "공식", "발표", "면접", "실습", "모임"] if w in text]
        st.write(", ".join(candidates) if candidates else "상황 키워드가 아직 뚜렷하지 않아요. 상황을 조금만 더 구체적으로 적어보세요!")

# =========================
# Footer
# =========================
st.divider()
st.caption(
    "안내: 본 앱은 진단/치료 목적이 아니며, 외모 평가를 하지 않습니다. "
    "추구미(이미지 정체성) 기준에서의 '적합성'만 다룹니다."
)
