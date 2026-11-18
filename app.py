"""AI-Powered Engineering Lab Companion: deterministic, offline-first demo."""
from io import BytesIO
import json
import importlib
import random
import re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import streamlit as st

from lab_companion import answer_question, classify_intent, generate_viva, retrieve, route_request
import gemini_viva
importlib.reload(gemini_viva)

GEMINI_KEY_FILE = Path(__file__).parent / ".streamlit" / "gemini_key.json"


def load_saved_gemini_key():
    try:
        return json.loads(GEMINI_KEY_FILE.read_text(encoding="utf-8")).get("api_key", "")
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""


def save_gemini_key(api_key):
    GEMINI_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    GEMINI_KEY_FILE.write_text(json.dumps({"api_key": api_key}), encoding="utf-8")


def forget_gemini_key():
    GEMINI_KEY_FILE.unlink(missing_ok=True)
    st.session_state["gemini_api_key_input"] = ""

st.set_page_config(page_title="Engineering Lab Companion", page_icon="⚙️", layout="wide")
st.title("⚙️ AI-Powered Engineering Lab Companion")
st.caption("Offline-first lab-document helper with a small ML intent classifier and a transparent tool router.")
@st.cache_data(show_spinner=False)
def _extract_uploaded_text_cached(name: str, content: bytes):
    """Extract once per uploaded file; Streamlit reuses the result on reruns."""
    if name.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        with ZipFile(BytesIO(content)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", ns):
            line = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
            if line:
                paragraphs.append(line)
        return "\n\n".join(paragraphs)
    raise ValueError("Supported formats are .txt, .md, .pdf, and .docx.")


def extract_uploaded_text(uploaded_file):
    return _extract_uploaded_text_cached(uploaded_file.name.lower(), uploaded_file.getvalue())

with st.sidebar:
    st.header("Lab and reference documents")
    uploaded_files = st.file_uploader(
        "Upload one or more handouts or viva sources (.txt, .md, .pdf, .docx)",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    default_path = Path(__file__).parent / "data" / "sample_lab_guide.txt"
    if uploaded_files:
        documents = []
        document_names = []
        for uploaded in uploaded_files:
            try:
                file_text = extract_uploaded_text(uploaded)
            except Exception as exc:
                st.error(f"Could not read {uploaded.name}: {exc}")
                st.stop()
            if not file_text.strip():
                st.warning(f"No selectable text found in {uploaded.name}; scanned PDFs need OCR and were skipped.")
                continue
            documents.append(f"# Document: {uploaded.name}\n\n{file_text}")
            document_names.append(uploaded.name)
        if not documents:
            st.error("No readable text was found in the uploaded files.")
            st.stop()
        document = "\n\n".join(documents)
        document_name = ", ".join(document_names)
    else:
        document = default_path.read_text(encoding="utf-8")
        document_name = "sample_lab_guide.txt"
    st.caption(f"Active sources: {document_name}")
    st.caption("PDFs need selectable text; scanned PDFs need OCR. Word .docx is supported (.doc is not). PDF equation layout may be distorted. The app searches only the files you upload, not the internet.")
st.header("Ask the lab guide")
question = st.text_input("Question", placeholder="What is the purpose of this experiment?")
if st.button("Ask", type="primary") and question.strip():
    intent = classify_intent(question)
    tool = route_request(intent)
    st.markdown(f"**Detected intent:** `{intent}`  ·  **Selected tool:** `{tool}`")
    result = answer_question(question, document)
    st.write(result["answer"])
    if result["sources"]:
        with st.expander("Retrieved evidence"):
            for item in result["sources"]:
                st.markdown(f"**{item['source']}** (similarity {item['score']:.2f})")
                st.write(item["text"])
    if result["warning"]:
        st.warning(result["warning"])

st.divider()
st.header("Agentic viva coach")
st.write("Create a real multiple-choice practice quiz. Offline mode uses your documents and built-in questions. Gemini mode can make four-option questions from broad course knowledge, with your manual as optional context.")
topic = st.text_input("Viva topic", placeholder="e.g., meter bridge, enthalpy of neutralisation, Experiment 3.2")
count = st.slider("Number of questions", 3, 8, 5)
use_gemini = st.checkbox("Generate AI viva questions with Gemini (optional; uses your API key)")
gemini_key = ""
privacy_ok = False
difficulty = "medium"
include_document = False
saved_gemini_key = load_saved_gemini_key()
remember_key = bool(saved_gemini_key)
if use_gemini:
    difficulty = st.selectbox("Quiz difficulty", ["beginner", "medium", "challenging"], index=1)
    include_document = st.checkbox("Use my uploaded manual as extra context (optional)", value=False)
    if saved_gemini_key:
        gemini_key = saved_gemini_key
        st.success("A Gemini key is saved on this computer and loaded automatically. The key entry box is hidden.")
        st.button("Forget saved API key", on_click=forget_gemini_key)
        st.caption("The key is stored locally in an unencrypted file. Do not share this project folder while it is saved.")
    else:
        gemini_key = st.text_input(
            "Google AI Studio API key",
            type="password",
            key="gemini_api_key_input",
            help="Keep this key private. Do not paste it into chat or share screenshots that reveal it.",
        )
        remember_key = st.checkbox(
            "Remember this key on this computer",
            help="Saves the key in a local file inside this project. The file is not encrypted; do not share the project folder while the key is saved.",
        )
    if include_document:
        st.warning("Gemini sends your topic and matching excerpts from the uploaded document to Google. Google’s free-tier terms say submitted content may be used to improve its products. Do not send private or confidential material. The model uses general knowledge too; it does not browse the web. No paid billing is enabled.")
        privacy_ok = st.checkbox("I understand and want to send the topic and manual excerpts to Google")
    else:
        st.info("Gemini will use general course knowledge, not your PDF. It does not perform a live web search. Check AI answers with your teacher or textbook.")
        privacy_ok = st.checkbox("I understand my topic will be sent to Google for question generation")
if "viva_items" not in st.session_state:
    st.session_state["viva_items"] = []
if "mock_viva_active" not in st.session_state:
    st.session_state["mock_viva_active"] = False
if "mock_viva_index" not in st.session_state:
    st.session_state["mock_viva_index"] = 0
if "mock_viva_revealed" not in st.session_state:
    st.session_state["mock_viva_revealed"] = False
st.session_state.setdefault("mock_viva_answers", {})


def make_mcq(item, document, question_number):
    """Use AI answer choices when present; otherwise build an offline evidence-selection quiz."""
    ai_options = item.get("options")
    answer_index = item.get("correct_index")
    if isinstance(ai_options, list) and len(ai_options) == 4 and isinstance(answer_index, int) and answer_index in range(4):
        choices = [(str(option), item.get("source", "AI-generated"), i == answer_index) for i, option in enumerate(ai_options)]
        return {"choices": choices, "correct_index": answer_index, "explanation": item.get("explanation", ""), "mode": "ai"}
    query = f"{item['question']} {item.get('source', '')}"
    matches = retrieve(query, document, limit=40)
    sentence_blocks = []
    seen = set()
    for match in matches:
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(match["text"].split())):
            sentence = sentence.strip(" -•")
            if len(sentence) < 35 or sentence.casefold() in seen:
                continue
            seen.add(sentence.casefold())
            sentence_blocks.append(f"### {match['source']}\n{sentence}")
    if len(sentence_blocks) < 4:
        return None
    ranked = retrieve(query, "\n\n".join(sentence_blocks), limit=40)
    passages, seen = [], set()
    for match in ranked:
        text = " ".join(match["text"].split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            passages.append({"text": text[:220].rstrip() + ("…" if len(text) > 220 else ""), "source": match["source"]})
        if len(passages) == 4:
            break
    if len(passages) < 4:
        return None
    choices = [(entry["text"], entry["source"], i == 0) for i, entry in enumerate(passages)]
    random.Random(f"{question_number}:{item['question']}").shuffle(choices)
    correct_index = next(i for i, choice in enumerate(choices) if choice[2])
    return {"choices": choices, "correct_index": correct_index, "explanation": "", "mode": "offline"}

if st.button("Generate viva questions"):
    try:
        if use_gemini:
            if not gemini_key.strip():
                st.error("Enter your Google AI Studio API key, or turn off Gemini to use offline questions.")
                items = []
            elif not privacy_ok:
                st.error("Please confirm the data-sharing note, or turn off Gemini to use offline questions.")
                items = []
            else:
                if remember_key:
                    save_gemini_key(gemini_key.strip())
                elif saved_gemini_key:
                    GEMINI_KEY_FILE.unlink(missing_ok=True)
                with st.spinner("Gemini is creating your viva questions and answer choices..."):
                    items = gemini_viva.generate_with_gemini(gemini_key, document, topic, count, difficulty, include_document)
        else:
            items = generate_viva(document, topic, count)
        if items:
            st.session_state["viva_items"] = items
            st.session_state["mock_viva_active"] = False
            st.session_state["mock_viva_index"] = 0
            st.session_state["mock_viva_revealed"] = False
            st.session_state["mock_viva_answers"] = {}
            for key in list(st.session_state.keys()):
                if key.startswith("mock_viva_submitted_"):
                    del st.session_state[key]
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info("You can turn off Gemini and generate built-in offline questions instead.")

viva_items = st.session_state["viva_items"]
if viva_items:
    with st.expander("View the full question list"):
        for i, item in enumerate(viva_items, 1):
            st.markdown(f"**Question {i}: {item['question']}**")
            st.caption(f"{item['source']} · {item['reason']}")

    if not st.session_state["mock_viva_active"]:
        if st.button("Start mock viva practice"):
            st.session_state["mock_viva_active"] = True
            st.session_state["mock_viva_index"] = 0
            st.session_state["mock_viva_revealed"] = False
            st.session_state["mock_viva_answers"] = {}
            for key in list(st.session_state.keys()):
                if key.startswith("mock_viva_submitted_"):
                    del st.session_state[key]
            st.rerun()

if st.session_state["mock_viva_active"] and viva_items:
    index = st.session_state["mock_viva_index"]
    if index >= len(viva_items):
        score = sum(st.session_state["mock_viva_answers"].values())
        total = len(viva_items)
        st.subheader("Mock viva complete")
        left, right = st.columns(2)
        left.metric("Your score", f"{score} / {total}")
        right.metric("Correct", f"{round(score / total * 100)}%" if total else "—")
        st.progress(score / total if total else 0, text="Review the cited passages for any missed questions.")
        if st.button("Try again", type="primary"):
            st.session_state["mock_viva_index"] = 0
            st.session_state["mock_viva_answers"] = {}
            for key in list(st.session_state.keys()):
                if key.startswith("mock_viva_submitted_"):
                    del st.session_state[key]
            st.rerun()
    else:
        item = viva_items[index]
        mcq = make_mcq(item, document, index)
        st.subheader("Mock viva · multiple choice")
        score = sum(st.session_state["mock_viva_answers"].values())
        c1, c2 = st.columns(2)
        c1.metric("Question", f"{index + 1} of {len(viva_items)}")
        c2.metric("Score so far", f"{score} / {len(st.session_state['mock_viva_answers'])}")
        st.progress((index + 1) / len(viva_items), text=f"Question {index + 1} of {len(viva_items)}")
        st.markdown(f"### {item['question']}")
        if mcq and mcq["mode"] == "ai":
            ai_context = " and uploaded manual context" if "uploaded manual context" in item.get("source", "") else ""
            st.caption(f"AI-generated viva MCQ · based on broad course knowledge{ai_context} · not a live web search")
        else:
            st.caption("Offline mode: choose the manual statement that best supports an answer. It uses your uploaded document, not general AI knowledge.")

        if mcq is None:
            st.warning("This document has fewer than four distinct statements to make four choices. Upload a longer handout or use Gemini to make a general-knowledge quiz.")
            if st.button("End practice"):
                st.session_state["mock_viva_active"] = False
                st.rerun()
        else:
            submitted_key = f"mock_viva_submitted_{index}"
            if not st.session_state.get(submitted_key, False):
                with st.form(f"mock_viva_form_{index}"):
                    labels = [f"{chr(65 + i)}. {choice[0]}" for i, choice in enumerate(mcq["choices"])]
                    selected = st.radio("Select one answer", labels)
                    submitted = st.form_submit_button("Check answer", type="primary")
                if submitted:
                    selected_index = labels.index(selected)
                    st.session_state["mock_viva_answers"][index] = selected_index == mcq["correct_index"]
                    st.session_state[submitted_key] = True
                    st.rerun()
            else:
                correct_index = mcq["correct_index"]
                correct_text, correct_source, _ = mcq["choices"][correct_index]
                if st.session_state["mock_viva_answers"].get(index, False):
                    st.success("Correct — that passage is the closest match to the question.")
                else:
                    st.error("Not quite. Review the passage that best matches the question.")
                if mcq.get("explanation"):
                    st.info(f"**Why:** {mcq['explanation']}")
                else:
                    st.info(f"**Best matching manual statement · {correct_source}:** {correct_text}")
                st.caption(f"Question source: {item['source']} · {item['reason']}")
                if st.button("Next question" if index + 1 < len(viva_items) else "See my result", type="primary"):
                    st.session_state["mock_viva_index"] += 1
                    st.rerun()

st.divider()
st.header("Model card: intent classifier")
st.write("A scikit-learn TF-IDF + Logistic Regression classifier predicts one of four intents: document question, viva practice, calculation, or safety. Its training set is a tiny hand-written demo; predictions can be wrong. The router maps each intent to a fixed local function, and shows that choice above.")
st.caption("This tool supports learning. It does not replace instructor guidance, lab rules, instrument manuals, or supervision. Verify every procedure and calculation before acting.")







