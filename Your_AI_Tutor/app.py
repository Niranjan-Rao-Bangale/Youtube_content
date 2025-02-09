# pip install google-generativeai streamlit openai numpy sympy matplotlib streamlit-drawable-canvas sentence-transformers chromadb
import base64
import uuid
from io import BytesIO

import chromadb
import google.generativeai as genai
import openai
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# -----------------------------
# Set Page Config (First Command)
# -----------------------------
st.set_page_config(page_title="Student AI Assistant", layout="wide")

# -----------------------------
# Initialize ChromaDB Client and Collection (outside any mode blocks)
# -----------------------------
persist_directory = "chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection_name = "student_questions"
question_collection = chroma_client.get_or_create_collection(name=collection_name)

# -----------------------------
# Initialize Sentence Embedding Model (outside any mode blocks)
# -----------------------------
embedding_model_name = "all-mpnet-base-v2"  # Good general-purpose sentence embedding model
embedding_model = SentenceTransformer(embedding_model_name)

# -----------------------------
# Initialize Session State
# -----------------------------
if "menu" not in st.session_state:
    st.session_state.menu = "Your Tutor"

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "ChatGPT"
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "learning_spaces" not in st.session_state:
    st.session_state.learning_spaces = ["Default Learning Space"]
if "learning_space_history" not in st.session_state:
    st.session_state.learning_space_history = {"Default Learning Space": []}
if "current_learning_space" not in st.session_state:
    st.session_state.current_learning_space = "Default Learning Space"
if "canvas_color" not in st.session_state:
    st.session_state.canvas_color = "#000000"
if "canvas_mode" not in st.session_state:
    st.session_state.canvas_mode = "freedraw"
if "canvas_tool" not in st.session_state:
    st.session_state.canvas_tool = "freedraw"
if "canvas_stroke_width" not in st.session_state:
    st.session_state.canvas_stroke_width = 3


def encode_image_to_base64(image):
    """Converts a PIL image to a Base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# -----------------------------
# Apply Theme and UI Styling
# -----------------------------
def apply_theme():
    if st.session_state.theme == "dark":
        dark_css = """
        <style>
        /* Dark Mode CSS - Comprehensive Styling */
        body,
        .stApp {
            background-color: #121212; /* Deeper dark background for body and app */
            color: #E0E0E0; /* Light grey text for better readability */
            font-weight: 400; /* Lighter font weight for body text in dark mode */
        }

        /* Sidebar Styles */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E; /* Darker sidebar background */
            color: #E0E0E0;
        }
        [data-testid="stSidebar"] .stButton>button {
            background-color: #272727 !important; /* Darker sidebar button background */
            color: #E0E0E0 !important;
            border-color: #424242 !important;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #333333 !important; /* Even darker on hover */
        }
        [data-testid="stSidebar"] a {
            color: #90CAF9 !important; /* Blue links in sidebar */
        }
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #E0E0E0; /* Expander header text in sidebar */
            font-weight: 600; /* Stronger weight for sidebar expander headers */
        }
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            color: #FAFAFA; /* Lighter on hover */
        }


        /* Main Content Styles */
        .block-container {
            background-color: transparent; /* Make block containers transparent */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FAFAFA; /* White-ish headings */
            font-weight: 700; /* Stronger weight for headings */
            letter-spacing: -0.02em; /* Slightly tighter letter spacing for headings */
        }
        p {
             font-weight: 400; /* Lighter font weight for paragraphs */
             line-height: 1.6; /* Improved line height for readability */
        }

        hr {
            border-color: #424242; /* Darker horizontal rule */
        }
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stNumberInput>div>div>input,
        .stDateInput>div>div>input,
        .stTimeInput>div>div>input,
        .stSelectbox>div>div>div,
        .stMultiselect>div>div>div,
        .stSlider>div>div>div,
        .stRadio>div>div,
        .stCheckbox>div>div,
        .stButton>button,
        .stDownloadButton>button {
            background-color: #272727 !important; /* Dark input/button background */
            color: #E0E0E0 !important;
            border-color: #424242 !important;
            font-weight: 500; /* Medium font weight for inputs/buttons */
        }
        .stButton>button {
            border: none !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.4) !important; /* More pronounced shadow */
            transition: background-color 0.2s, transform 0.1s; /* Add transform for button press effect */
        }
        .stButton>button:hover {
            background-color: #333333 !important;
        }
        .stButton>button:active {
            transform: scale(0.98); /* Button press effect */
        }

        .stApp a {
            color: #90CAF9 !important; /* Blue links in main content */
            text-decoration: none; /* Remove underlines from links if desired */
        }
        .stApp a:hover {
            text-decoration: underline; /* Underline on hover for links */
        }

        .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
            background-color: #272727 !important; /* Darker alert backgrounds */
            color: #E0E0E0 !important;
            border-color: #424242 !important;
            border-radius: 0.4rem; /* Rounded corners for alerts */
            padding: 1rem !important; /* Padding for alerts */
            margin-bottom: 1rem !important; /* Margin below alerts */
        }
        .streamlit-expander {
            background-color: #1E1E1E !important; /* Dark expander background */
            border-color: #424242 !important;
            border-radius: 0.4rem !important; /* Rounded corners for expanders */
            margin-bottom: 1rem !important;
        }
        .streamlit-expanderHeader {
            font-weight: 600 !important; /* Stronger font for expander header */
            padding: 0.7rem !important; /* Padding for expander header */
        }


        .streamlit-expanderContent {
            background-color: #272727 !important; /* Dark expander content background */
            padding: 1rem !important;
            border-radius: 0 0 0.4rem 0.4rem; /* Rounded bottom corners for content */
        }


        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
    elif st.session_state.theme == "light":
        light_css = """
        <style>
        /* Light Mode CSS - Comprehensive Styling */
        body,
        .stApp {
            background-color: #F5F5F5; /* Light grey background for body and app */
            color: #333; /* Darker text for light mode */
            font-weight: 400; /* Standard font weight for body text in light mode */
        }
        /* Sidebar Styles */
        [data-testid="stSidebar"] {
            background-color: #EEEEEE; /* Lighter sidebar background */
            color: #333;
        }
        [data-testid="stSidebar"] .stButton>button {
            background-color: #E0E0E0 !important; /* Lighter sidebar button background */
            color: #333 !important;
            border-color: #BDBDBD !important;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #D5D5D5 !important; /* Slightly darker on hover */
        }
        [data-testid="stSidebar"] a {
            color: #1976D2 !important; /* Darker blue links in sidebar */
        }
         [data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #333; /* Expander header text in sidebar */
            font-weight: 600; /* Stronger weight for sidebar expander headers */
        }
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            color: #555; /* Darker on hover */
        }

        /* Main Content Styles */
        .block-container {
            background-color: transparent; /* Make block containers transparent */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #212121; /* Darker headings */
            font-weight: 700; /* Stronger weight for headings */
             letter-spacing: -0.02em;
        }
         p {
             font-weight: 400;
             line-height: 1.6;
        }
        hr {
            border-color: #BDBDBD; /* Lighter horizontal rule */
        }
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stNumberInput>div>div>input,
        .stDateInput>div>div>input,
        .stTimeInput>div>div>input,
        .stSelectbox>div>div>div,
        .stMultiselect>div>div>div,
        .stSlider>div>div>div,
        .stRadio>div>div,
        .stCheckbox>div>div,
        .stButton>button,
        .stDownloadButton>button {
            background-color: #FFFFFF !important; /* White input/button background */
            color: #333 !important;
            border-color: #BDBDBD !important;
             font-weight: 500;
        }
        .stButton>button {
            border: none !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important; /* Subtle shadow */
             transition: background-color 0.2s, transform 0.1s; /* Add transform for button press effect */
        }
        .stButton>button:hover {
            background-color: #EEEEEE !important;
        }
         .stButton>button:active {
            transform: scale(0.98); /* Button press effect */
        }

        .stApp a {
            color: #1976D2 !important; /* Darker blue links in main content */
            text-decoration: none;
        }
         .stApp a:hover {
            text-decoration: underline;
        }
        .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
            background-color: #E0E0E0 !important; /* Lighter alert backgrounds */
            color: #333 !important;
            border-color: #BDBDBD !important;
             border-radius: 0.4rem;
             padding: 1rem !important;
             margin-bottom: 1rem !important;
        }
        .streamlit-expander {
            background-color: #EEEEEE !important; /* Lighter expander background */
            border-color: #BDBDBD !important;
             border-radius: 0.4rem !important;
             margin-bottom: 1rem !important;
        }
        .streamlit-expanderHeader {
             font-weight: 600 !important;
             padding: 0.7rem !important;
        }
        .streamlit-expanderContent {
            background-color: #F5F5F5 !important; /* Lighter expander content background */
             padding: 1rem !important;
             border-radius: 0 0 0.4rem 0.4rem;
        }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

    # Apply global styles (independent of theme)
    global_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        h1, h2, h3, h4, h5, h6, p, div, body, span, a, button, input, textarea, select {
            font-family: 'Roboto', sans-serif !important;
        }
        .stApp {
            max-width: 90%;
            padding-left: 1rem;
            padding-right: 1rem;
            margin: 0 auto;
        }
        /* Futuristic UI Enhancements (Global) */
        .stButton>button {
            border-radius: 10px !important; /* More rounded buttons */
            letter-spacing: 0.05em; /* Slight letter spacing for buttons */
            transition: all 0.2s ease-in-out; /* Smooth transitions for all button properties */
        }
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stNumberInput>div>div>input,
        .stDateInput>div>div>input,
        .stTimeInput>div>div>input,
        .stSelectbox>div>div>div,
        .stMultiselect>div>div>div,
        .stSlider>div>div>div,
        .stRadio>div>div,
        .stCheckbox>div>div {
            border-radius: 8px !important; /* Rounded corners for inputs */
            padding: 0.7rem 1rem !important; /* Slightly larger padding for inputs */
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.06); /* Subtle inner shadow for depth */
        }
        .st-e721ln, .block-container { /* Target sections and containers for more spacing */
            padding-top: 2.5rem !important; /* Increased padding */
            padding-bottom: 2.5rem !important;
            margin-bottom: 1.5rem !important; /* Increased margin */
        }
        h1 {
            font-size: 2.5rem !important; /* Larger main heading */
        }
        h2 {
            font-size: 1.8rem !important; /* Slightly larger subheadings */
        }


        </style>
        """
    st.markdown(global_css, unsafe_allow_html=True)

apply_theme()

# -----------------------------
# Sidebar UI - Clickable Navigation and Canvas Tools (Conditional)
# -----------------------------
with st.sidebar:
    st.title("Hi, Niranjan 👋")
    st.markdown("### Welcome to your AI Tutor 🚀")

    if st.button("📚 Your Tutor", key="tutor_btn", icon="🎓"):
        st.session_state.menu = "Your Tutor"
        st.rerun()
    if st.button("💬 Ask AI", key="ask_btn", icon="❓"):
        st.session_state.menu = "Ask"
        st.rerun()
    if st.button("📝 Canvas", key="canvas_btn", icon="🎨"):
        st.session_state.menu = "Canvas"
        st.rerun()
    if st.button("📚 Spaces", key="search_btn", icon="📂"):
        st.session_state.menu = "Search"
        st.rerun()
    if st.button("➕ New Space", key="add_space_btn", icon="➕"):
        st.session_state.menu = "Add Space"
        st.rerun()
    if st.button("⚙️ Settings", key="settings_btn", icon="⚙️"):
        st.session_state.menu = "Settings"
        st.rerun()

    st.markdown("---")
    st.subheader("Spaces:")
    for space in st.session_state.learning_spaces:
        if st.button(f"📁 {space}", key=f"space_button_{space}"):
            st.session_state.current_learning_space = space
            st.session_state.menu = "Search"
            st.rerun()

    # --------------------------------------------------
    # Canvas Tools - Conditionally shown in Sidebar ONLY in Canvas Mode
    # --------------------------------------------------
    if st.session_state.menu == "Canvas":
        st.markdown("---") # Separator above canvas tools
        with st.expander("Canvas Tools", expanded=True):
            st.subheader("Drawing Tools")
            # Tool selection buttons
            tool_col1, tool_col2 = st.columns(2)
            with tool_col1:
                if st.button("Free Draw", key="freedraw_tool_btn", use_container_width=True):
                    st.session_state.canvas_tool = "freedraw"
            with tool_col2:
                if st.button("Eraser", key="eraser_tool_btn", use_container_width=True):
                    st.session_state.canvas_tool = "eraser"

            st.subheader("Shapes")
            shape_col1, shape_col2 = st.columns(2)
            with shape_col1:
                 if st.button("Line", key="line_tool_btn", use_container_width=True):
                    st.session_state.canvas_tool = "line"
                 if st.button("Rect", key="rect_tool_btn", use_container_width=True):
                    st.session_state.canvas_tool = "rect"
            with shape_col2:
                if st.button("Circle", key="circle_tool_btn", use_container_width=True):
                    st.session_state.canvas_tool = "circle"
                if st.button("Transform", key="transform_tool_btn", use_container_width=True):  # Pan/zoom tool
                    st.session_state.canvas_tool = "transform"

            st.subheader("Color & Stroke")
            st.color_picker("Stroke Color:", key="canvas_color")  # Color picker
            st.number_input("Stroke Width:", min_value=1, max_value=20, value=st.session_state.canvas_stroke_width,
                            key="canvas_stroke_width")  # Stroke width control


# -----------------------------
# Mode 1: Your Tutor (Main Screen) - Landing Page Redesign
# -----------------------------
if st.session_state.menu == "Your Tutor":
    st.title("🎓 Welcome to Your AI Tutor")
    st.markdown("Engage with interactive learning:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬 Ask a Question", key="tutor_ask_btn", use_container_width=True, icon="❓"):
            st.session_state.menu = "Ask"
            st.rerun()

    with col2:
        if st.button("📝 Open Canvas", key="tutor_canvas_btn", use_container_width=True, icon="🎨"):
            st.session_state.menu = "Canvas"
            st.rerun()

    st.markdown("---")
    st.markdown("Explore learning spaces and settings in the sidebar.")


# -----------------------------
# Mode 2: Ask (Chat with AI)
# -----------------------------
elif st.session_state.menu == "Ask":
    st.subheader(f"💡 Ask in **{st.session_state.current_learning_space}**")

    # --- Check LLM configuration before proceeding ---
    llm_provider = st.session_state.get("llm_provider")
    openai_api_key = st.session_state.get("openai_api_key")
    gemini_api_key = st.session_state.get("gemini_api_key")

    if llm_provider is None or llm_provider == "None":
        st.warning("Please select an LLM provider in the Settings menu to use this feature.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop execution if no provider selected
    elif llm_provider == "ChatGPT" and not openai_api_key:
        st.warning("Please enter your LLM API key in the Settings menu to use LLM.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop if OpenAI API key is missing for ChatGPT
    elif llm_provider == "Gemini" and not gemini_api_key:
        st.warning("Please enter your Gemini API key in the Settings menu to use Gemini.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop if Gemini API key is missing for Gemini
    # --- End LLM configuration check ---

    st.warning("This AI tutor can make mistakes. Double-check important answers.", icon="⚠️")

    # Display chat history for current learning space
    if st.session_state.learning_space_history[st.session_state.current_learning_space]:
        st.markdown("#### Chat History:")
        for chat in st.session_state.learning_space_history[st.session_state.current_learning_space]:
            st.markdown(f"**{chat['role']}:** {chat['message']}")
        if st.button("Clear History", key="clear_history_btn"):
            st.session_state.learning_space_history[st.session_state.current_learning_space] = []
            st.rerun()

    question = st.text_input("Enter your question:")

    if st.button("Get Answer", key="ask_submit_btn"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.llm_provider == "ChatGPT":
                        openai.api_key = st.session_state.openai_api_key
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system",
                                     "content": "You are an AI tutor. "
                                                "Provide educational explanations in simple language."},
                                    {"role": "user", "content": question}
                                ],
                                temperature=0.7
                            )
                            answer = response.choices[0].message.content.strip()
                        except Exception as e:
                            answer = f"Error with ChatGPT API: {e}"  # More informative error
                            st.error(answer)

                    elif st.session_state.llm_provider == "Gemini":
                        genai.configure(api_key=st.session_state.gemini_api_key)  # Set Gemini API key
                        model = genai.GenerativeModel('gemini-2.0-flash')  # For text-only

                        try:
                            response = model.generate_content(question)
                            answer = response.text
                            if not answer:  # Handle cases where Gemini might return empty response
                                answer = ("Gemini returned an empty response. "
                                          "Please try again or rephrase your question.")
                        except Exception as e:
                            answer = f"Error with Gemini API: {e}"  # More informative error
                            st.error(answer)

                    else:
                        answer = "LLM Provider not selected or recognized."
                        st.error(answer)

                    # Store history in the CURRENT learning space
                    st.session_state.learning_space_history[st.session_state.current_learning_space].append(
                        {"role": "Student", "message": question})
                    st.session_state.learning_space_history[st.session_state.current_learning_space].append(
                        {"role": "AI Tutor", "message": answer})

                    st.markdown(f"**AI Tutor:** {answer}")

                    question_embedding = embedding_model.encode(question).tolist()  # Generate embedding
                    question_collection.add(
                        embeddings=[question_embedding],  # Embeddings are vectors of floats
                        documents=[question],  # Original question text
                        ids=[str(uuid.uuid4())]  # Generate unique ID for each question (using uuid - import uuid)
                    )

                except Exception as e:
                    st.error(f"Error getting answer from AI. Please check API key and connection. Error: {e}")

# -----------------------------
# Mode 3: Open Canvas with AI Analysis - Chat Interface Integrated
# -----------------------------
elif st.session_state.menu == "Canvas":
    st.title("🖊️ Interactive Drawing Canvas")
    # --- Check LLM configuration before proceeding ---
    llm_provider = st.session_state.get("llm_provider")
    openai_api_key = st.session_state.get("openai_api_key")
    gemini_api_key = st.session_state.get("gemini_api_key")

    if not llm_provider:
        st.warning("Please select an LLM provider in the Settings menu to use this feature.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop execution if no provider selected
    elif llm_provider == "ChatGPT" and not openai_api_key:
        st.warning("Please enter your LLM API key in the Settings menu to use LLM.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop if OpenAI API key is missing for ChatGPT
    elif llm_provider == "Gemini" and not gemini_api_key:
        st.warning("Please enter your Gemini API key in the Settings menu to use Gemini.", icon="⚠️")
        if st.button("Go to Settings", key="go_to_settings_btn_from_warning_no_provider"):  # Unique key
            st.session_state.menu = "Settings"
            st.rerun()
        st.stop()  # Stop if Gemini API key is missing for Gemini
    # --- End LLM configuration check ---
    st.markdown(f"Space: **{st.session_state.current_learning_space}**")
    st.warning("AI analysis of drawings may not be perfect. Double-check conclusions.", icon="⚠️")
    st.write("Draw diagrams or math expressions using the tools below: **Icons represent Free Draw, Eraser, "
             "Shapes, Color Picker, and Stroke Width**. This is a simplified canvas...")

    canvas_col, chat_col = st.columns([0.7, 0.3])

    with canvas_col:
        # ------------------------- Canvas Toolbar (Icon-based) -------------------------
        st.markdown("##### Tools:") # Toolbar label
        col_tools1, col_tools2, col_tools3, col_tools4, col_tools5, col_tools6 = st.columns(6)
        with col_tools1:
            if st.button("✏️", key="canvas_toolbar_freedraw_btn", use_container_width=True, help="Free Draw"):
                st.session_state.canvas_tool = "freedraw"
        with col_tools2:
            if st.button("⚪", key="canvas_test_eraser", use_container_width=True, help="Eraser"):
                st.session_state.canvas_tool = "eraser"
        with col_tools3:
            if st.button("➖", key="canvas_toolbar_line_btn", use_container_width=True, help="Line"):
                st.session_state.canvas_tool = "line"
        with col_tools4:
            if st.button("⬛", key="canvas_toolbar_rect_btn", use_container_width=True, help="Rectangle"):
                st.session_state.canvas_tool = "rect"
        with col_tools5:
            if st.button("⚪", key="canvas_toolbar_circle_btn", use_container_width=True, help="Circle"):
                st.session_state.canvas_tool = "circle"
        with col_tools6:
            if st.button("🖐️", key="canvas_toolbar_transform_btn", use_container_width=True, help="Transform/Pan"):
                st.session_state.canvas_tool = "transform"

        st.markdown("---") # Separator line after tools

        # Color Picker and Stroke Width (Still using widgets, placed after toolbar for now)
        st.subheader("Color & Stroke")
        if st.session_state.canvas_tool == "eraser":
            actual_drawing_mode = "freedraw"
            actual_stroke_color = "#FFFFFF"  # same color as background
        else:
            actual_drawing_mode = st.session_state.canvas_tool
            actual_stroke_color = st.session_state.canvas_color

        from streamlit_drawable_canvas import st_canvas

        uploaded_image_file = st.file_uploader("Upload image for analysis:", type=["png", "jpg", "jpeg"],
                                               key="canvas_image_uploader")
        background_image = None  # Initialize background_image to None
        if uploaded_image_file is not None:
            background_image = Image.open(uploaded_image_file).convert("RGB")
            # encoded_image = encode_image_to_base64(image)
            # background_image = f"data:image/png;base64,{encoded_image}"

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=st.session_state.get("canvas_stroke_width", 3),
            stroke_color=actual_stroke_color,
            background_color="#ffffff",
            background_image=background_image if uploaded_image_file is not None else None,
            height=400,
            width=600,
            drawing_mode=actual_drawing_mode,
            key="canvas_draw"
        )

    with st.container():
        st.subheader("Canvas Chat")

        # Display chat history for current learning space in Canvas context
        canvas_chat_history = [chat for chat in st.session_state.learning_space_history[
            st.session_state.current_learning_space] if chat.get("context") == "canvas"]
        if canvas_chat_history:
            st.markdown("#### Chat History:")
            for chat in canvas_chat_history:
                st.markdown(f"**{chat['role']}:** {chat['message']}")
            if st.button("Clear Canvas Chat History", key="clear_canvas_history_btn"):
                st.session_state.learning_space_history[st.session_state.current_learning_space] =\
                    [chat for chat in st.session_state.learning_space_history[st.session_state.current_learning_space]
                     if chat.get("context") != "canvas"] # Clear only canvas chats
                st.rerun()
            st.markdown("---")

        question = st.text_input("Ask about your drawing:", key="canvas_question_input")
        if st.button("Ask AI about Canvas", key="ask_canvas_ai_btn"):
            if question:
                    with st.spinner("Analyzing drawing and thinking..."):
                        try:
                            if st.session_state.llm_provider == "ChatGPT":
                                image_analysis = openai.chat.completions.create(
                                    model="gpt-4o-mini",  # USE THIS UPDATED MODEL NAME
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": question},  # Text part of the content
                                                {
                                                    "type": "image_url",  # Image part of content
                                                    "image_url": {
                                                        "url": "data:image/png;base64," + base64.b64encode(
                                                            open("canvas_input.png", "rb").read()).decode()
                                                    },
                                                },
                                            ],
                                        },
                                    ],
                                    max_tokens=300
                                )
                                answer = image_analysis.choices[0].message.content.strip()

                            elif st.session_state.llm_provider == "Gemini":
                                import google.generativeai as genai

                                genai.configure(api_key=st.session_state.gemini_api_key)  # Set Gemini API Key
                                model = genai.GenerativeModel('gemini-2.0-flash')  # For vision input
                                try:
                                    image_part = {"mime_type": "image/png", "data": base64.b64encode(
                                        open("canvas_input.png", "rb").read()).decode()}  # Prepare image data
                                    response = model.generate_content(
                                        [question, image_part])  # Gemini takes content as list
                                    answer = response.text
                                    if not answer:  # Handle empty response
                                        answer = ("Gemini returned an empty response. Please try again or rephrase "
                                                  "your question.")
                                except Exception as e:
                                    answer = f"Error with Gemini API: {e}"
                                    st.error(answer)

                            else:
                                answer = "LLM Provider not selected or recognized."
                                st.error(answer)
                            st.session_state.learning_space_history[st.session_state.current_learning_space].\
                                append({"role": "Student (Canvas)", "message": question, "context": "canvas"})
                            st.session_state.learning_space_history[st.session_state.current_learning_space].\
                                append({"role": "AI Tutor (Canvas)", "message": answer, "context": "canvas"})

                            st.markdown(f"**AI Tutor (Canvas):** {answer}")

                        except Exception as e:
                            st.error(f"Error analyzing drawing. Please check API key and connection. Error: {e}")

# -----------------------------
# Mode 4: Search Spaces (Now Showing Learning Spaces and History)
# -----------------------------
elif st.session_state.menu == "Search":
    st.title("📚 Learning Spaces")
    st.markdown("Click on a learning space in the sidebar to view its history and details.")

    if st.session_state.learning_spaces:
        # Display learning spaces as expanders - history is inside expander
        for space in st.session_state.learning_spaces:
            with st.expander(f"📁 **{space}**", expanded=st.session_state.current_learning_space == space):
                st.subheader("Chat History")
                if st.session_state.learning_space_history[space]:
                    for chat in st.session_state.learning_space_history[space]:
                        st.markdown(f"**{chat['role']}:** {chat['message']}")
                else:
                    st.info("No chat history yet for this learning space.", icon="ℹ️")
    else:
        st.info("You haven't created any learning spaces yet. "
                "Click 'Add Space' in the sidebar to get started!", icon="ℹ️")


# -----------------------------
# Mode 5: Add Learning Space
# -----------------------------
elif st.session_state.menu == "Add Space":
    st.title("➕ Create New Space")
    space_name = st.text_input("Enter a name for this learning space:", key="new_space_input")

    if st.button("Create Space", key="create_space_btn"):
        if space_name:
            st.session_state.learning_spaces.append(space_name)
            st.session_state.learning_space_history[space_name] = []
            st.session_state.current_learning_space = space_name
            st.success(f"Learning space **{space_name}** created and set as current space.")
            st.rerun()

# -----------------------------
# Mode 6: Settings (Theme Selection - Radio Button)
# -----------------------------
elif st.session_state.menu == "Settings":
    st.title("⚙️ Settings")
    st.subheader("LLM Provider")
    llm_provider = st.selectbox(
        "Select LLM Provider:",
        options=["Gemini", "ChatGPT"],
        index=0,
        key="llm_provider_selectbox"
    )
    st.session_state.llm_provider = llm_provider
    if st.session_state.llm_provider == "ChatGPT":
        openai_api_key = st.text_input("Enter OpenAI API Key:", type="password", key="openai_api_key_input",
                                       value=st.session_state.get("openai_api_key", ""))  # Persist value if available
        st.session_state.openai_api_key = openai_api_key  # Store OpenAI API Key
    elif st.session_state.llm_provider == "Gemini":
        gemini_api_key = st.text_input("Enter Gemini API Key:", type="password", key="gemini_api_key_input",
                                       value=st.session_state.get("gemini_api_key", ""))  # Persist value if available
        st.session_state.gemini_api_key = gemini_api_key  # Store Gemini API Key

    st.warning(
        "⚠️ API keys are sensitive. For a real application, use secure methods to manage API keys "
        "(like Streamlit Secrets) instead of plain text inputs.", icon="⚠️")

    st.subheader("Theme")

    theme_choice = st.radio(
        "Select Theme:",
        options=["Light Mode", "Dark Mode"],
        index=0 if st.session_state.theme == "light" else 1
    )

    if theme_choice == "Light Mode":
        st.session_state.theme = "light"
    elif theme_choice == "Dark Mode":
        st.session_state.theme = "dark"

    apply_theme()
    st.write(f"Current Theme: **{st.session_state.theme.capitalize()} Mode**")


# -----------------------------
# Mode 7: Exit to Home
# -----------------------------
elif st.session_state.menu == "Home":
    st.title("🏠 Home") # Shortened title - "Exiting" implied by Home icon now
    st.write("Welcome to the Home Screen. Use the sidebar to navigate.") # More appropriate message for home
