# SwarVarta: AI Framework for Real-Time Indian Language and Dialect Intelligence 🇮🇳

## 📌 Project Overview
SwarVarta is an AI-powered chatbot framework designed to handle **real-time conversations in Indian languages and dialects**.  
It integrates **custom-trained intent detection models** with **Google Gemini API** to provide accurate, context-aware responses.

The framework focuses on:
- Understanding **multilingual and code-mixed conversations**.
- Supporting **music playback** and **study/tutorial video recommendations**.
- Offering a **chat interface** for smooth interaction.
- Bringing **AI-powered conversational intelligence** to rural and urban India.

---

## ✅ Features Implemented Till Date
1. **Intent Detection Model**
   - Trained model for recognizing intents like `play_music`, `study_video`, `general_question`, etc.
   - Label encoder and preprocessing pipeline integrated.

2. **FastAPI Backend (`main.py`)**
   - Exposes a `/predict` API endpoint.
   - Handles text input, predicts intent, and routes query accordingly.
   - Connects to **Gemini API** for general question answering.

3. **Web Search Integration**
   - For `play_music` → Autoplays YouTube music.
   - For `study_video` → Fetches tutorial/educational content from YouTube.
   - For `general_question` → Uses Gemini model to answer.

4. **Frontend Chat UI (`index.html`)**
   - Simple chat interface with:
     - User & bot messages
     - Autoplay YouTube video/music inside chat
     - Project description displayed at the top

5. **Basic Testing Completed**
   - API tested via Swagger (`/docs`).
   - End-to-end chatbot tested locally.

---

## 🚀 Next Steps
- Enhance **multilingual NLP** support (Hindi, Marathi, Tamil, etc.).
- Integrate **speech-to-text** and **text-to-speech** for full voice conversation.
- Deploy chatbot online (using free hosting like **Render / Railway**).
- Improve **UI/UX** with a modern design and voice input button.
- Add **pause/stop controls** for music/video playback.

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI (Python)
- **AI/ML**: TensorFlow/Keras for intent model, Gemini API for generative answers
- **Media Search**: YouTube Search + Autoplay
- **Hosting (Planned)**: Free-tier cloud platforms

---

## 👩‍💻 Contributors
- **Janhavi Katakdhond** – Project Lead & Developer
- **Mansi Mahabdi** - Developer

---

## Running
- **HTML Document** -  python -m http.server 5500
- **main.py file** - unicorn main:app --reload
