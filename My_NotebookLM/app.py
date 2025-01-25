import os

from flask import Flask, render_template, request, jsonify

from my_notebooklm import build_qa_system

app = Flask(__name__)

# Global variable to store the QA system
qa_system = None

@app.route("/")
def home():
    return render_template("index.html")


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/upload", methods=["POST"])
def upload_files():
    global qa_system
    
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    files = request.files.getlist("files")
    urls = request.form.getlist("urls")
    
    if not files and not urls:
        return jsonify({"message": "Please upload at least one file or provide a URL.", "status": "error"})

    file_paths = []
    if files:
        for file in files:
            if file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                file_paths.append(filepath)

    try:
        qa_system = build_qa_system("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", file_paths, urls) # replase XXXXXXXXXXXXXXXXXXXXXXXXXXXX with your gemini api key
        return jsonify({"message": "QA system built successfully!", "status": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"})


@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_system
    if not qa_system:
        return jsonify({"message": "Please upload files or provide URLs to initialize the system.", "status": "error"})
    
    question = request.form.get("question")
    try:
        raw_answer  = qa_system.invoke({"query": question})
        if isinstance(raw_answer, dict) and "result" in raw_answer:
            final_answer = raw_answer["result"]
        else:
            final_answer = str(raw_answer)
        return jsonify({"question": question, "answer": final_answer, "status": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"})



if __name__ == "__main__":
    app.run(debug=True)
