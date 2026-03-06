Perfect — you’re ready to finalize your project 🎯

Here’s a complete README.md written in a clean, professional format for your ModelForge AI project.
It explains setup, folder structure, commands, features, and troubleshooting — perfect for GitHub or documentation.

# 🧠 ModelForge AI
**A no-code intelligent AI model builder** — upload any dataset, train classification/chatbot/knowledge models, and export ready-to-use models — all through an interactive web UI.

---

## 🚀 Overview

**ModelForge AI** is an end-to-end system that lets users:
- Upload datasets (CSV, JSON, TXT)
- Automatically preprocess and clean data
- Train machine learning models (SVM, Logistic Regression, Random Forest, LLMs)
- Visualize training metrics
- Download the trained model as a `.zip` package
- (Optional) Deploy or test within the browser

It combines:
- 🧩 **FastAPI backend** for model training and orchestration  
- 💻 **React + TypeScript frontend** for a modern, guided UI workflow  

---
  
## 🗂️ Project Structure



modelforgeai/
│
├── api_server.py # 🚀 FastAPI backend server
├── core/ # 🧠 Core logic modules
│ ├── dataset_handler.py # Data loading and cleaning
│ ├── model_trainer.py # Model training logic
│ ├── model_manager.py # Metadata + model packaging
│ ├── prompt_parser.py # Task auto-detection helper
│ ├── visualizer.py # Accuracy & metric plots
│ └── ...
│
├── datasets/ # 📂 Uploaded and processed datasets
│ ├── classification/
│ ├── processed/
│
├── models/ # 🧠 Trained model files
│ └── classification/
│
├── outputs/ # 📦 Packaged model zips and charts
│ └── packages/
│
├── frontend/ # 🌐 React + TypeScript UI
│ ├── src/
│ │ ├── pages/
│ │ │ └── CreateModel.tsx # Main frontend flow
│ │ └── lib/api.ts # API connection layer
│ └── ...
│
└── README.md # 📘 You’re here


---

## ⚙️ Installation

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/yourusername/modelforgeai.git
cd modelforgeai

2️⃣ Backend Setup (FastAPI)
🔹 Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate    # (Windows)
# or
source venv/bin/activate # (Mac/Linux)

🔹 Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, you can create one with:

pip install fastapi uvicorn scikit-learn pandas numpy joblib matplotlib
pip freeze > requirements.txt

🔹 Run the backend
uvicorn api_server:app --reload


You should see:

INFO:     Uvicorn running on http://127.0.0.1:8000


✅ Check:

API Docs → http://127.0.0.1:8000/docs

Health Check → http://127.0.0.1:8000/health

3️⃣ Frontend Setup (React + Vite)

Move into the frontend folder:

cd frontend


Install dependencies:

npm install


Run the dev server:

npm run dev


You’ll see:

Local: http://127.0.0.1:5173/


✅ Open your browser → http://127.0.0.1:5173

🧭 How to Use

Select Task Type
Choose between:

Classification – train NLP or sentiment models

Chatbot – build FAQ/chat engines

Knowledge – create embeddings for RAG

Choose Model Type
Auto-selects best algorithm (or manually pick SVM, Random Forest, etc.)

Upload Dataset
Upload .csv, .json, or .txt. The backend auto-cleans and preprocesses it.

Set Preprocessing Options
Adjust train/test split or column names.

Train Model
Watch real-time training progress (handled by FastAPI background jobs).

Visualize & Download
Once done, a .zip is generated with:

Trained model

Metadata

Performance charts

Training logs

Test / Deploy
Try sample predictions or export model for deployment.

🔄 API Endpoints Summary
Endpoint	Method	Description
/	GET	Root status
/health	GET	Backend health check
/tasks	GET	Task options
/models	GET	Model options
/datasets/upload	POST	Upload and preprocess dataset
/preprocess	POST	Set test split
/train/start	POST	Begin background training job
/train/status/{job_id}	GET	Poll training status
/download/{filename}	GET	Download packaged model
🧰 Tech Stack
Frontend

⚛️ React (Vite + TypeScript)

🎨 TailwindCSS + ShadCN UI

🔔 Lucide Icons

🌈 Toast notifications & Progress UI

Backend

⚡ FastAPI

🤖 Scikit-learn, Pandas, NumPy

📈 Matplotlib for charts

💾 Joblib for model storage

📦 Zip packaging for downloads

🧩 CORS Setup

Since the frontend (port 5173) communicates with backend (port 8000),
CORS is enabled in api_server.py:

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

⚠️ Common Issues
Problem	Fix
UserWarning: Field "model_type" has conflict with protected namespace	Add BaseModel.model_config = {"protected_namespaces": ()} under your app creation
CORS errors	Ensure allow_origins=["*"] or add "http://127.0.0.1:5173"
Attribute "app" not found	Run as uvicorn api_server:app --reload (not main)
Frontend can’t connect	Make sure both servers are running (backend: 8000, frontend: 5173)
🏁 Example Run

Start backend
uvicorn api_server:app --reload

Start frontend
npm run dev

Open http://127.0.0.1:5173

Follow on-screen steps:

Select → Task Type

Upload → Dataset

Train → Model

Download → .zip file

🧾 License

MIT License © 2025
Developed by [ GADI GURU SAGAR REDDY]

🌟 Future Improvements

Support for GPU/LLM fine-tuning

Model performance dashboards

One-click cloud deployment

Auto column detection for datasets

Project save/load workflows
