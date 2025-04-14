import datetime
import json
import os
import uuid
from typing import List
from typing import Optional

import openai
from fastapi import FastAPI, Form
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

openai.api_key = "<Your_Open_AI_Key"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores
job_profiles = {}
candidate_scores = {}


class JobProfile(BaseModel):
    title: str
    skills: List[str]
    experience: str
    question_types: List[str]


class CodeSubmission(BaseModel):
    code: str


# For logging interview sessions
interview_sessions = {}
MAX_QUESTIONS = 3  # Configurable number of voice turns


# Existing job profile endpoints omitted for brevity…
# (submit_job_profile, get_uploaded_file, etc.)

@app.get("/interview/start")
async def start_interview(job_id: str, candidate_name: str):
    # Retrieve job profile (either from memory or .txt file)
    profile = job_profiles.get(job_id)
    if not profile:
        txt_file_path = os.path.join("uploads", f"{job_id}.txt")
        if os.path.exists(txt_file_path):
            try:
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {txt_file_path}: {e}")
                return JSONResponse(status_code=500, content={"error": "Failed to load job profile"})
        else:
            return JSONResponse(status_code=404, content={"error": "Job profile not found"})

    # Ensure profile is a dict with structured keys (for manual submissions)
    if isinstance(profile, dict):
        title = profile.get("Job Title", "a suitable role")
        raw_skills = profile.get("Skills", [])
        if isinstance(raw_skills, str):
            try:
                raw_skills = json.loads(raw_skills)
            except json.JSONDecodeError:
                raw_skills = []
        skills = ", ".join(raw_skills)
    else:
        print(f"Unexpected profile format: {profile}")
        return JSONResponse(status_code=500, content={"error": "Invalid job profile format"})

    if skills:
        intro_text = (
            f"Hello {candidate_name}, welcome to your interview for the position of {title}. "
            f"Please start by introducing yourself. I’ll also ask questions about your experience in {skills}."
        )
    else:
        intro_text = (
            f"Hello {candidate_name}, welcome to your interview. "
            "Please begin by introducing yourself. I will ask questions based on your uploaded profile."
        )

    # Create a unique session and log file for this interview
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    log_path = f"interview_logs/{job_id}_{candidate_name}_{timestamp_str}.json"
    os.makedirs("interview_logs", exist_ok=True)
    interview_sessions[(job_id, candidate_name)] = {"log": log_path, "turn": 1}
    print(f"Interview session created for {(job_id, candidate_name)}: {log_path}")

    with open(log_path, "w") as f:
        json.dump([{"role": "assistant", "text": intro_text}], f, indent=2)

    # Generate TTS audio for the intro text
    filename = f"static/intro_{timestamp_str}.mp3"
    open_ai_voice_client(intro_text, filename)

    # Log the intro on the backend console
    print("Intro sent:", intro_text)

    return {"text": intro_text, "audio_url": f"/{filename}"}


@app.post("/job_profile/submit/")
async def submit_job_profile(
    mode: str = Form(...),
    title: Optional[str] = Form(None),
    skills: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    question_types: Optional[str] = Form(None),
    profile_file: Optional[UploadFile] = File(None)
):
    job_id = str(uuid.uuid4())
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if mode == "file":
        if profile_file is None:
            return {"error": "File upload missing"}
        ext = os.path.splitext(profile_file.filename)[1]
        filename = f"{job_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            content = await profile_file.read()
            f.write(content)
        job_profiles[job_id] = {
            "file_name": filename,
            "file_path": file_path
        }

    elif mode == "manual":
        filename = f"{job_id}.txt"
        file_path = os.path.join(UPLOAD_DIR, filename)
        profile_text = {"Job Title": title, "Skills": skills, "Experience": experience, "Question Types": question_types}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profile_text, f, indent=4)

        job_profiles[job_id] = {
            "title": title,
            "skills": skills,
            "experience": experience,
            "question_types": question_types,
            "file_name": filename,
            "file_path": file_path
        }

    else:
        return {"error": "Invalid mode"}

    return {"message": "Job profile saved successfully.", "job_id": job_id}

@app.post("/interview/evaluate")
async def evaluate_response(
        candidate_text: str = Form(...),
        job_id: str = Form(...),
        candidate_name: str = Form(...),
        max_questions: int = 3
):
    # Print candidate response for logging
    print(f"Candidate response from {candidate_name} (Job ID: {job_id}): {candidate_text}")

    # Retrieve session based on job_id and candidate_name
    session_key = (job_id, candidate_name)
    if session_key not in interview_sessions:
        return JSONResponse(
            status_code=400,
            content={"error": "Interview session not found. Please start the interview first."}
        )
    session = interview_sessions[session_key]
    log_path = session["log"]
    turn = session["turn"]

    # Load conversation log
    with open(log_path, "r") as f:
        conversation = json.load(f)
    conversation.append({"role": "candidate", "text": candidate_text})
    print("Conversation so far:", conversation)

    # If we have not reached the max number of voice questions, generate next voice question
    if turn < max_questions:
        system_prompt = "You are an AI interviewer. Ask the next question based on the candidate's last response."
        messages = [{"role": "system", "content": system_prompt}]
        for entry in conversation:
            role = "user" if entry["role"] == "candidate" else "assistant"
            messages.append({"role": role, "content": entry["text"]})
        gpt_response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        next_question = gpt_response.choices[0].message.content
        session["turn"] += 1
    else:
        # Max voice questions reached: conclude voice phase and ask coding question
        system_prompt = "You are an AI interviewer. Please ask a basic Python coding interview question."
        gpt_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}]
        )
        coding_question = gpt_response.choices[0].message.content
        next_question = (
                "Analyzing your response. I will provide your feedback to the project manager and HR. "
                "Now, please solve the following coding problem. "
                + coding_question +
                "\nSelect your programming language and write your solution below."
        )
        session["turn"] += 1

    conversation.append({"role": "assistant", "text": next_question})
    with open(log_path, "w") as f:
        json.dump(conversation, f, indent=2)

    # Generate TTS for "Analyzing your response..." and the next question
    analyzing_path = f"static/analyzing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    open_ai_voice_client("Analyzing your response...", analyzing_path)
    question_path = f"static/question_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    open_ai_voice_client(next_question, question_path)

    # Log full conversation on the backend
    print("Updated conversation log:", conversation)

    # Return response; include a flag if coding phase is active
    letCodingPhase = session["turn"] > max_questions
    return {
        "feedback": candidate_text,
        "next_question": next_question,
        "analyzing_audio": f"/{analyzing_path}",
        "question_audio": f"/{question_path}",
        "coding": letCodingPhase,
        "conversation_log": conversation  # For debugging
    }


@app.post("/interview/evaluate_code")
async def evaluate_code_response(
        candidate_code: str = Form(...),
        language: str = Form(...),
        job_id: str = Form(...),
        candidate_name: str = Form(...)
):
    # Log candidate code for backend review
    print(f"Candidate {candidate_name} submitted code (Language: {language}) for Job ID: {job_id}:")
    print(candidate_code)

    # Use job profile context if needed (optional)
    profile = job_profiles.get(job_id, {})
    context_info = f"Job Title: {profile.get('title', 'N/A')}, Skills: {profile.get('skills', 'N/A')}"

    system_prompt = (
        "You are an AI interviewer evaluating a coding interview response. "
        f"Candidate submitted the following {language} code in response to a coding question. "
        f"Context: {context_info}. Please provide feedback on correctness, efficiency, and style, "
        "and give a score out of 10."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": candidate_code}
    ]

    gpt_response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    feedback = gpt_response.choices[0].message.content
    # Log GPT evaluation on backend
    print("GPT evaluation for candidate code:", feedback)

    return {"feedback": feedback}


@app.get("/candidate")
def candidate_ui():
    return FileResponse("static/candidate.html")


def open_ai_voice_client(text, audio_file_path, speed=0.8):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="sage",
        input=text,
        speed=speed
    )
    response.stream_to_file(audio_file_path)
