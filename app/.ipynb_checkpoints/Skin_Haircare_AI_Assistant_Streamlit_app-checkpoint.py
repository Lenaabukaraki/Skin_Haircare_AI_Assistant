#!/usr/bin/env python3
messages = list(st.session_state.chat_history) # copy
if st.session_state.uploaded_image:
last_ia = next((r for r in reversed(st.session_state.records) if r.get("type") == "image_analysis" and r.get("summary")), None)
if last_ia:
messages.append({"role": "user", "content": f"Image analysis context: {last_ia['summary']}"})
messages.append({"role": "user", "content": user_text})


with st.spinner("Contacting chat model..."):
result = hf_chat_completion(messages, model_hint=selected_chat_model)


if result.get("ok"):
# Update state
st.session_state.chat_history.append({"role": "user", "content": user_text})
reply = result["reply"]
st.session_state.chat_history.append({"role": "assistant", "content": reply})
# Show typewriter
ph = st.empty()
typewriter_markdown(ph, f"**Assistant:** {reply}")
# Record
st.session_state.records.append({
"type": "chat_turn",
"model": result.get("model"),
"user": user_text,
"assistant": reply,
})
# TTS (gTTS)
try:
from gtts import gTTS # lazy import
tts = gTTS(reply)
mp3_path = "reply.mp3"
tts.save(mp3_path)
with open(mp3_path, "rb") as f:
st.audio(f.read(), format="audio/mp3")
except Exception as e:
st.info("Text-to-speech unavailable right now.")
else:
st.error(result.get("error", "Unknown error during chat."))


st.divider()


# Chat history
with st.expander("💬 Conversation history", expanded=True):
for msg in st.session_state.chat_history:
role = msg["role"].capitalize()
st.markdown(f"**{role}:** {msg['content']}")


# Export
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