# Skin_Haircare_AI_Assistant

A privacy‑minded Streamlit app for basic skin/hair image analysis (local model) + chat guidance (Hugging Face Inference). Includes CLI tools to test multiple models locally or via the remote API. **Not medical advice.**

## Features
- 🔍 **Local** image analysis with `WahajRaza/finetuned-dermnet` (keeps images on your machine)
- 💬 **Chat** with fallback models on HF Inference (env token required)
- 🎙️ Voice input via mic or audio upload → Whisper transcription
- 🔊 gTTS voice replies
- 📦 Export chat + predictions as JSON
- 🧰 CLI scripts to compare models (local & remote)

## Quick Start
```bash
# 1) Clone
git clone https://github.com/<your-username>/Skin_Haircare_AI_Assistant.git
cd Skin_Haircare_AI_Assistant

# 2) (optional) venv & deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Set your Hugging Face token (chat/ASR)
export HF_API_TOKEN="hf_..."   # Windows PowerShell:  setx HF_API_TOKEN "hf_..."

# 4) Run the app
streamlit run app/Skin_Haircare_AI_Assistant.py
