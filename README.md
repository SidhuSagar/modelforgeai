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
- Test predictions within the browser

It combines:
- 🧩 **FastAPI backend** for model training and orchestration  
- 💻 **React + TypeScript frontend** for a modern, guided UI workflow  

---

## 🗂️ Project Structure

```
modelforgeai/
│
├── backend/              # 🐍 Python backend code
│   ├── api/             # API routers and schemas
│   ├── core/            # Core logic modules
│   │   ├── dataset_handler.py
│   │   ├── model_trainer.py
│   │   ├── model_manager.py
│   │   ├── predictor.py
│   │   └── visualizer.py
│   ├── services/        # Business logic services
│   ├── configs/         # Configuration files
│   ├── scripts/         # Utility scripts
│   ├── tests/           # Test files
│   └── api_server.py    # FastAPI application entry point
│
├── frontend/            # ⚛️ React + TypeScript frontend
│   ├── src/
│   │   ├── pages/       # Page components
│   │   ├── components/  # Reusable components
│   │   └── lib/         # Utilities and API client
│   └── package.json
│
├── docs/                # 📚 Documentation
├── requirements.txt     # Python dependencies
├── Dockerfile          # Backend Docker image
├── docker-compose.yml     # Docker orchestration
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/modelforgeai.git
cd modelforgeai
```

### 2️⃣ Backend Setup (FastAPI)

**Create and activate a virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the backend:**

```bash
cd backend
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Note:** Make sure to run from the `backend/` directory so that relative imports work correctly.

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

✅ **Check:**
- API Docs → http://127.0.0.1:8000/docs
- Health Check → http://127.0.0.1:8000/health

### 3️⃣ Frontend Setup (React + Vite)

**Move into the frontend folder:**

```bash
cd frontend
```

**Install dependencies:**

```bash
npm install
```

**Run the dev server:**

```bash
npm run dev
```

You'll see:
```
Local: http://127.0.0.1:5173/
```

✅ **Open your browser → http://127.0.0.1:5173**

---

## 🐳 Docker Setup (Alternative)

**Using Docker Compose:**

```bash
docker-compose up --build
```

This will start both backend (port 8000) and frontend (port 5173) services.

**Backend only:**

```bash
docker build -t modelforgeai-backend .
docker run -p 8000:8000 modelforgeai-backend
```

---

## 🧭 How to Use

1. **Select Task Type**
   - Choose between: Classification, Chatbot, or Knowledge base

2. **Choose Model Type**
   - Auto-selects best algorithm (or manually pick SVM, Random Forest, etc.)

3. **Upload Dataset**
   - Upload `.csv`, `.json`, or `.txt`. The backend auto-cleans and preprocesses it.

4. **Set Preprocessing Options**
   - Adjust train/test split or column names.

5. **Train Model**
   - Watch real-time training progress (handled by FastAPI background jobs).

6. **Visualize & Download**
   - Once done, a `.zip` is generated with:
     - Trained model
     - Metadata
     - Performance charts
     - Training logs

7. **Test / Deploy**
   - Try sample predictions or export model for deployment.

---

## 🔄 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root status |
| `/health` | GET | Backend health check |
| `/tasks` | GET | Task options |
| `/models` | GET | Model options |
| `/datasets/upload` | POST | Upload and preprocess dataset |
| `/preprocess` | POST | Set test split |
| `/train/start` | POST | Begin background training job |
| `/train/status/{job_id}` | GET | Poll training status |
| `/download/{filename}` | GET | Download packaged model |

Full API documentation available at: http://127.0.0.1:8000/docs

---

## 🧰 Tech Stack

### Frontend
- ⚛️ React (Vite + TypeScript)
- 🎨 TailwindCSS + ShadCN UI
- 🔔 Lucide Icons
- 🌈 Toast notifications & Progress UI

### Backend
- ⚡ FastAPI
- 🤖 Scikit-learn, Pandas, NumPy
- 📈 Matplotlib for charts
- 💾 Joblib for model storage
- 📦 Zip packaging for downloads
- 🔮 Sentence Transformers (optional)
- 🚀 Transformers / PyTorch (optional)

---

## 🧩 CORS Setup

Since the frontend (port 5173) communicates with backend (port 8000), CORS is enabled in `backend/api_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ⚠️ Common Issues

| Problem | Fix |
|---------|-----|
| CORS errors | Ensure `allow_origins=["*"]` or add frontend URL |
| Attribute "app" not found | Run as `uvicorn api_server:app --reload` from `backend/` directory |
| Frontend can't connect | Make sure both servers are running (backend: 8000, frontend: 5173) |
| Import errors | Ensure you're running from the correct directory and virtual environment is activated |

---

## 🏁 Quick Start Example

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn api_server:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open browser:**
```
http://127.0.0.1:5173
```

Follow on-screen steps:
1. Select → Task Type
2. Upload → Dataset
3. Train → Model
4. Download → `.zip` file

---

## 📝 Development

### Running Tests

```bash
cd backend
python -m pytest tests/
```

### Code Structure

- **Backend**: Follows FastAPI best practices with modular core logic
- **Frontend**: Component-based React architecture with TypeScript
- **API**: RESTful endpoints with background job processing

---

## 🧾 License

MIT License © 2025

Developed by [GADI GURU SAGAR REDDY]

---

## 🌟 Future Improvements

- [ ] Support for GPU/LLM fine-tuning
- [ ] Model performance dashboards
- [ ] One-click cloud deployment
- [ ] Auto column detection for datasets
- [ ] Project save/load workflows
- [ ] Model versioning and registry
- [ ] Real-time collaboration features

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.
