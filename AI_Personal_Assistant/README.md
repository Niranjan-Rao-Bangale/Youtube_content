# AI Personal Assistant
An AI-powered personal assistant that integrates with **Google Calendar & Gmail** to help manage emails, schedule meetings, and respond to invitations automatically.

---

## 🚀 Features
### ✅ Email Management
- Reads **unread emails** (excluding spam/promotions)
- Generates **AI-based replies** using OpenAI's GPT
- Allows users to **edit and send AI-generated responses**

### ✅ Meeting Invitation Handling
- Detects and extracts **Google Calendar meeting invites** from emails
- Enables users to **Accept, Decline, or Propose a New Time**
- Updates **Google Calendar RSVP** status automatically

### ✅ Google Calendar Integration
- Fetches **scheduled meetings**
- Syncs **RSVP status** with Google Calendar

### ✅ User Authentication
- Uses **Google OAuth 2.0** for secure authentication
- Supports multiple Google services (**Gmail, Calendar, People API**)

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Niranjan-Rao-Bangale/Youtube_content.git
cd Youtube_content/AI_Personal_Assistant
```

## **2️⃣Install Dependencies**
Ensure you have Python 3.8+ installed.
```sh
pip install -r requirements.txt
```

### **3️⃣ Set Up Google API Credentials**
- Go to Google Cloud Console
- Enable the following APIs:
  - Gmail API
  - Google Calendar API
  - People API (for user profile details)
- Download your credentials.json from Google Cloud Console
- Place it in the project directory (AI_Personal_Assistant/)

### **4️⃣ Run the Application**
```sh
streamlit run main.py
```

### **⚙️ Configuration**
```python
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/contacts.readonly",
    "openid"
]
```
OAuth Notes:

The first authentication will open a browser for login.
If issues occur, delete token.pickle and retry.

### **🛠 Troubleshooting**
### **❌ OAuth Authentication Issues**
Solution:

Ensure you have the correct OAuth credentials.
Verify that the APIs are enabled in Google Cloud Console.
Run:
```sh
rm token.pickle
streamlit run main.py
```
### **❌ Encoding Issues in AWS SageMaker**
If running in AWS SageMaker and facing encoding errors:

Convert file encoding before processing:
```ssh
iconv -f WINDOWS-1252 -t UTF-8 reports.csv -o reports_utf8.csv
```

### **🤝 Contributing**
Pull requests are welcome! If you have suggestions, feel free to open an issue.

### **📜 License**
MIT License - Feel free to use and modify!

This README.md file provides complete details on setup, usage, and troubleshooting. Let me know if you'd like any modifications! 🚀
