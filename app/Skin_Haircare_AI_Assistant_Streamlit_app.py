#!/usr/bin/env python3
"""
Skin & Haircare AI Assistant (Streamlit)
- Local image analysis (privacy-friendly)
- Chat via HF Inference with model fallbacks
- Optional voice input (mic or audio upload)
- Typewriter effect and JSON export
- No hard-coded tokens; uses HF_API_TOKEN
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline
import torch
import requests

# Optional: load .env for local dev (no-op if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "🧴 Skin & Haircare AI Assistant"
DISCLAIMER = (
    "**Important:** This app provides educational guidance only and is **not** a "
    "medical diagnosis. If you notice rapidly changing, bleeding, or concerning "
    "lesions, please consult a dermatologist."
)

# Chat models to try (order matters)
CHAT_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Whisper model for ASR (HF Inference)
ASR_MODEL = "openai/whisper-small"

# Local image classifier
IMG_MODEL = "WahajRaza/finetuned-dermnet"

# HF token (required for chat + ASR)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HAS_TOKEN = bool(HF_API_TOKEN and HF_API_TOKEN.startswith("hf_"))

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.info(DISCLAIMER)

with st.expander("ℹ️ Privacy & data flow", expanded=False):
    st.markdown(
        "- **Local**: Image analysis runs locally (first run downloads model weights to cache).  \n"
        "- **Remote**: Chat replies and optional speech transcription use Hugging Face Inference APIs (requires `HF_API_TOKEN`).  \n"
        "- Avoid sharing personal medical identifiers in chat prompts."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Load/cached resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_classifier():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=IMG_MODEL, device=device)

classifier = load_classifier()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with expertise in skincare and haircare. "
                "Answer clearly, professionally, and safely."
            ),
        }
    ]

if "records" not in st.session_state:
    st.session_state.records: List[Dict[str, Any]] = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image: Optional[Image.Image] = None

# Sidebar
st.sidebar.header("Settings")
chosen_model = st.sidebar.selectbox("Chat model (fallback-ready)", CHAT_MODELS, index=0)
use_typewriter = st.sidebar.toggle("Typewriter effect", value=True)
if not HAS_TOKEN:
    st.sidebar.warning("HF_API_TOKEN is not set — chat and ASR will be disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def analyze_skin_image(pil_img: Image.Image) -> Dict[str, Any]:
    """Run local image classification and return a structured dict."""
    try:
        results = classifier(pil_img, top_k=3)
        if not results:
            return {"ok": False, "error": "No predictions returned."}
        best = results[0]
        summary = f"Possible issue: **{best['label']}** ({best['score']:.2%})."
        return {"ok": True, "summary": summary, "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def hf_chat_completion(messages: List[Dict[str, str]], model_hint: Optional[str] = None) -> Dict[str, Any]:
    """Call HF Inference chat with fallbacks; returns {ok, reply, model} or {ok:False, error}."""
    if not HAS_TOKEN:
        return {"ok": False, "error": "HF_API_TOKEN missing; cannot call remote chat."}

    from huggingface_hub import InferenceClient  # lazy import
    client = InferenceClient(api_key=HF_API_TOKEN)

    tried: List[str] = []
    models_to_try = [model_hint] + CHAT_MODELS if model_hint else CHAT_MODELS

    for m in models_to_try:
        if not m or m in tried:
            continue
        tried.append(m)
        # Try chat.completions API
        try:
            resp = client.chat.completions.create(model=m, messages=messages)
            reply = resp.choices[0].message.content  # OpenAI-style response
            return {"ok": True, "model": m, "reply": reply}
        except Exception:
            # Fall back to text_generation by packing messages into a prompt
            try:
                prompt = "\n".join([f"{x['role'].upper()}: {x['content']}" for x in messages]) + "\nASSISTANT:"
                tg = client.text_generation(prompt=prompt, model=m, max_new_tokens=512)
                return {"ok": True, "model": m, "reply": tg}
            except Exception:
                continue

    return {"ok": False, "error": f"All chat models failed: {', '.join(tried)}"}

def hf_asr_transcribe(audio_bytes: bytes) -> Dict[str, Any]:
    """Transcribe audio using HF Inference Whisper; returns {ok, text} or {ok:False, error}."""
    if not HAS_TOKEN:
        return {"ok": False, "error": "HF_API_TOKEN missing; cannot transcribe."}
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{ASR_MODEL}",
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=audio_bytes,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "text" in data:
            return {"ok": True, "text": data["text"]}
        if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
            return {"ok": True, "text": data[0]["text"]}
        return {"ok": False, "error": f"Unexpected ASR response: {data}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def typewriter_markdown(container, text: str, speed: float = 0.02):
    if not text:
        return
    if not use_typewriter:
        container.markdown(text)
        return
    shown = ""
    for ch in text:
        shown += ch
        container.markdown(shown + "**|**")
        time.sleep(speed)
    container.markdown(shown)

# ─────────────────────────────────────────────────────────────────────────────
# Image upload & analysis
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Image upload (optional)")
img_file = st.file_uploader("Upload a skin/hair image", type=["jpg", "jpeg", "png"], key="img_up")

if img_file:
    try:
        img = Image.open(img_file)
        img = ImageOps.exif_transpose(img).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        st.session_state.uploaded_image = img

        with st.spinner("Analyzing locally..."):
            analysis = analyze_skin_image(img)

        if analysis.get("ok"):
            st.success(analysis["summary"])
        else:
            st.warning(f"Image analysis failed: {analysis.get('error', 'unknown error')}")

        st.session_state.records.append({
            "type": "image_analysis",
            "summary": analysis.get("summary"),
            "raw": analysis.get("results"),
        })
    except Exception as e:
        st.error(f"Could not process image: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Voice input (optional)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Ask by voice (optional)")
voice_col1, voice_col2 = st.columns(2)
voice_text: Optional[str] = None

with voice_col1:
    used_mic = False
    try:
        # Optional dependency: streamlit-mic-recorder
        from streamlit_mic_recorder import mic_recorder  # type: ignore

        audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop", key="mic")
        if audio and isinstance(audio, dict) and audio.get("bytes"):
            used_mic = True
            st.toast("Transcribing...")
            asr = hf_asr_transcribe(audio["bytes"])  # mic_recorder returns WAV/PCM bytes
            if asr.get("ok"):
                voice_text = asr["text"]
                st.success(f"You said: {voice_text}")
            else:
                st.warning(f"ASR error: {asr.get('error')}")
    except Exception:
        # Component not installed or other error; ignored
        pass

with voice_col2:
    up = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg"], key="audio_up")
    if up and not voice_text:
        try:
            wav_bytes = up.read()
            asr = hf_asr_transcribe(wav_bytes)
            if asr.get("ok"):
                voice_text = asr["text"]
                st.success(f"Transcribed: {voice_text}")
            else:
                st.warning(f"ASR error: {asr.get('error')}")
        except Exception as e:
            st.warning(f"Failed to transcribe: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Chat input + send
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Ask anything")
q_col1, q_col2 = st.columns([5, 1])

with q_col1:
    default_text = voice_text or ""
    user_text = st.text_input("Your question", value=default_text, label_visibility="collapsed")
with q_col2:
    send = st.button("Send", use_container_width=True)

if send and user_text.strip():
    # Build messages; include latest image analysis summary if present
    messages = list(st.session_state.chat_history)
    last_ia = None
    if "records" in st.session_state:
        last_ia = next(
            (r for r in reversed(st.session_state.records)
             if r.get("type") == "image_analysis" and r.get("summary")),
            None
        )
    if last_ia:
        messages.append({"role": "user", "content": f"Image analysis context: {last_ia['summary']}"})
    messages.append({"role": "user", "content": user_text})

    with st.spinner("Contacting chat model..."):
        result = hf_chat_completion(messages, model_hint=chosen_model)

    if result.get("ok"):
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        reply = result["reply"]
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        ph = st.empty()
        typewriter_markdown(ph, f"**Assistant:** {reply}")

        st.session_state.records.append({
            "type": "chat_turn",
            "model": result.get("model"),
            "user": user_text,
            "assistant": reply,
        })

        # Optional TTS (gTTS)
        try:
            from gtts import gTTS  # lazy import
            tts = gTTS(reply)
            mp3_path = "reply.mp3"
            tts.save(mp3_path)
            with open(mp3_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        except Exception:
            st.info("Text-to-speech unavailable right now.")
    else:
        st.error(result.get("error", "Unknown error during chat."))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# History + Export
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("💬 Conversation history", expanded=True):
    for msg in st.session_state.chat_history:
        role = msg["role"].capitalize()
        st.markdown(f"**{role}:** {msg['content']}")

st.subheader("Export session")
export_data = {
    "image_model": IMG_MODEL,
    "chat_models": CHAT_MODELS,
    "records": st.session_state.records,
}
export_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button(
    label="Download chat + predictions (JSON)",
    data=export_bytes,
    file_name="session_export.json",
    mime="application/json",
)
