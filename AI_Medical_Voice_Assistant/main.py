#pip install fastapi uvicorn openai google-cloud-texttospeech

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import torch
from TTS.api import TTS
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
openai.api_key = "sk-proj-your_OPEN_AI_API_KEY"


class UserInput(BaseModel):
    text: str


def tts_client(text, audio_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(TTS().list_models())
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(
        text=text,
        speaker_wav="C:/Users/bnira/PycharmProjects/TTS/AI_Assistant/female-vocal-321-countdown-240912.mp3",
        language="en",
        file_path=audio_file_path,
        emotion="joy")


def open_ai_voice_client(text, audio_file_path, speed=0.8):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="sage",
        input=text,
        speed=speed
    )
    response.stream_to_file(audio_file_path)


@app.post("/chat/")
async def chat(user_input: UserInput, model="gpt-4", temperature=0.7):
    print(f"Received input: {user_input.text}")
    system_prompt = """
    You're name is Siri, you're a AI Heal Consultant. You are trained to provide basic health advice, 
    information about appointment booking, and support services. Follow these guidelines:

    - If the user asks about symptoms, provide general advice but remind them to consult a doctor for an official diagnosis.
    - If the user wants to book or cancel an appointment, ask for the date, time, and doctor’s name if needed.
    - If the user inquires about prescriptions, explain that they should contact their pharmacy or physician for refills.
    - If the user asks about billing or insurance, provide general guidelines and suggest contacting the hospital billing department for specifics.
    - Keep responses concise, polite, and professional.
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_input.text}]
        )
        ai_response = response.choices[0].message.content
        # tts_client(ai_response, "static/response.mp3")
        current_timestamp = int(time.time())
        open_ai_voice_client(ai_response, f"static/response_{current_timestamp}.mp3")
        print(f"AI response: {ai_response}")
        print(f"Generated audio file: static/response_{current_timestamp}.mp3")
        return {"text": ai_response, "audio_url": f"/static/response_{current_timestamp}.mp3"}
    except Exception as e:
        print(f"Error in processing request: {e}")
        return {"text": "Error processing request.", "audio_url": ""}


@app.get("/")
async def root():
    return StaticFiles(directory="static", html=True)
