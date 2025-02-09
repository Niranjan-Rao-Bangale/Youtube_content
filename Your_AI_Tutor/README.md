Your AI Tutor - Your Personalized Learning Assistant 🚀

## Overview

Your AI Tutor is an interactive Streamlit application designed to enhance learning through personalized AI assistance.  It offers a multi-faceted approach to education, combining the power of large language models with an interactive drawing canvas and organized learning spaces. Whether you need quick answers to complex questions, visual learning through diagrams, or a structured environment to study, Your AI Tutor has you covered.

### Key Features:

#### Ask AI Tutor: 
Get instant, simplified explanations to your questions powered by state-of-the-art AI models (ChatGPT & Gemini).
####Interactive Drawing Canvas: 
Visualize concepts by drawing diagrams, math expressions, or uploading images. Ask the AI tutor questions about your canvas creations or uploaded images for deeper understanding.
####LLM Provider Choice: 
Select between ChatGPT and Gemini as your AI engine and easily input your own API keys for flexibility and control.
####Learning Spaces: 
Organize your learning by creating dedicated spaces for different subjects or projects, keeping your chat history and learning materials organized.
####Theme Customization: 
Switch between Light Mode and Dark Mode for a comfortable learning experience in any environment.
####Persistent Chat History: 
Your chat conversations are saved and loaded across sessions, allowing you to review past interactions within each learning space.
####Tech Stack

This application is built using the following Python libraries:

Streamlit - For creating the interactive web application.
OpenAI Python Library - To interface with OpenAI's ChatGPT models.
google-generativeai - To interface with Google's Gemini models.
NumPy - For numerical operations.
SymPy - For symbolic mathematics (potentially used in future enhancements).
Matplotlib - For plotting and visualization (potentially used in future enhancements).
streamlit-drawable-canvas - For the interactive drawing canvas component.
sentence-transformers - For generating sentence embeddings for question similarity (used for future enhancements).
ChromaDB - For vector database to store and search embeddings (used for future enhancements).
Pillow (PIL) - Python Imaging Library for image processing.
Setup and Installation
Follow these steps to get the Your AI Tutor application running on your local machine:

#### Prerequisites:

Python 3.8 or higher - Make sure you have Python installed on your system. You can download it from python.org.
pip - Python package installer (usually comes with Python installations).
Installation Steps:

Clone the repository (if you have access to the code repository):

```
git clone [repository_url]
cd [your_repository_directory]
Create a virtual environment (recommended):
```

```
python -m venv .venv
Activate the virtual environment:
```
On Windows:

```
.venv\Scripts\activate
```
On macOS/Linux:
```
source .venv/bin/activate
```

Install required Python libraries:

Create a requirements.txt file in the same directory as your app.py with the following content:

```
streamlit
openai
numpy
sympy
matplotlib
streamlit-drawable-canvas
sentence-transformers
chromadb
google-generativeai
Pillow
```

Then, install the dependencies using pip:

Bash
```
pip install -r requirements.txt
```

Set up API Keys:

OpenAI API Key (for ChatGPT): You will need an OpenAI API key to use ChatGPT. You can obtain one from OpenAI's website.
Gemini API Key (for Gemini): You will need a Google Gemini API key to use Gemini. You can obtain one from Google AI Studio.
Important:  For security, it is recommended to use Streamlit Secrets to manage your API keys in a production environment. However, for local testing, you can directly input the API keys in the "Settings" section of the application.

Run the Streamlit application:

```
streamlit run app.py
```

This command will start the Streamlit server, and the application will be accessible in your web browser, usually at http://localhost:8501.

#### How to Use
##### Navigation: 
Use the sidebar menu on the left to navigate between different modes:

###### Your Tutor: 
Landing page with quick access to key features.
###### Ask AI: 
Chat interface to ask questions and get AI-powered answers.
###### Canvas: 
Interactive drawing canvas with AI analysis capabilities.
###### Spaces: 
Manage and access your learning spaces and chat history.
###### Add Space: 
Create new learning spaces to organize your study.
###### Settings: 
Configure application settings, including theme and LLM provider/API keys.
###### Settings:

* Before using the "Ask AI" or "Canvas" features, go to the "Settings" menu.
* Select your preferred LLM Provider (ChatGPT or Gemini).
* Enter the corresponding API Key for the selected provider.
* Choose your preferred Theme (Light or Dark Mode).

###### Ask AI:

* Navigate to "Ask AI" from the sidebar.
* Select your current Learning Space (or use the default).
* Type your question in the "Enter your question" text input.
* Click the "Get Answer" button to receive an AI-powered response.
* Review the chat history displayed above the input.

######Canvas:

* Navigate to "Canvas" from the sidebar.
* Use the toolbar to select drawing tools: Free Draw, Eraser, Line, Rectangle, Circle, Transform/Pan.
* Adjust Color & Stroke Width using the controls below the toolbar.
* Draw on the canvas using the selected tools.
* Optionally, upload an image using the "Upload image for analysis:" file uploader to use as a background or to ask questions about.
* Type your question about your drawing or uploaded image in the "Ask about your drawing:" text input below the canvas.
* Click "Ask AI about Canvas" to get AI analysis and explanations.
Review the chat history for canvas interactions above the input.

###### Learning Spaces:

* Manage existing learning spaces and create new ones using the sidebar menu.
* Select a learning space to view its chat history in the "Spaces" mode.
