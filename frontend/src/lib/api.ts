// src/lib/api.ts
const API_BASE = "http://127.0.0.1:8000";

export const api = {
  // 1️⃣ Get available tasks
  async getTasks(): Promise<any> {
    const res = await fetch(`${API_BASE}/tasks`);
    if (!res.ok) throw new Error(`Failed to get tasks: ${res.status}`);
    return res.json();
  },

  // 2️⃣ Get available model types (auto, logisticregression, etc.)
  async getModels(): Promise<any> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`Failed to get models: ${res.status}`);
    return res.json();
  },

  // 3️⃣ Upload dataset file
  async uploadDataset(taskType: string, file: File): Promise<any> {
    const formData = new FormData();
    formData.append("task_type", taskType);
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`Dataset upload failed: ${res.status}`);
    return res.json();
  },

  // 4️⃣ Set preprocessing options
  async setPreprocessing(testSplit: number): Promise<any> {
    const formData = new FormData();
    formData.append("test_split", testSplit.toString());

    const res = await fetch(`${API_BASE}/preprocess`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`Preprocessing setup failed: ${res.status}`);
    return res.json();
  },

  // 5️⃣ Start training job
  async startTraining({
    taskType,
    modelType,
    datasetPath,
    testSplit,
    epochs,
  }: {
    taskType: string;
    modelType: string;
    datasetPath: string;
    testSplit: number;
    epochs: number;
  }): Promise<any> {
    const formData = new FormData();
    formData.append("task_type", taskType);
    formData.append("model_type", modelType);
    formData.append("dataset_path", datasetPath);
    formData.append("test_split", testSplit.toString());
    formData.append("epochs", epochs.toString());

    const res = await fetch(`${API_BASE}/train/start`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`Training start failed: ${res.status}`);
    return res.json();
  },

  // 6️⃣ Get training job status
  async checkStatus(jobId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/train/status/${jobId}`);
    if (!res.ok) throw new Error(`Failed to check status: ${res.status}`);
    return res.json();
  },

  // 7️⃣ Download model file (ZIP)
  getDownloadUrl(filename: string): string {
    return `${API_BASE}/download/${filename}`;
  },

  // -------------------------
  // NEW: Model management & prediction endpoints
  // -------------------------

  /**
   * List available model files on server (most recent first)
   */
  async listModels(): Promise<any> {
    const res = await fetch(`${API_BASE}/models/list`);
    if (!res.ok) throw new Error(`Failed to list models: ${res.status}`);
    return res.json();
  },

  /**
   * Load a model into server memory (cached Predictor).
   * Backend expects FormData with field "model_path".
   */
  async loadModel(modelPath: string): Promise<any> {
    const form = new FormData();
    form.append("model_path", modelPath);

    const res = await fetch(`${API_BASE}/models/load`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      // backend returns HTTP 400 with detail on failure
      const text = await res.text();
      throw new Error(`Failed to load model: ${res.status} ${text}`);
    }
    return res.json();
  },

  /**
   * Get info about currently loaded model (if any)
   */
  async getCurrentModel(): Promise<any> {
    const res = await fetch(`${API_BASE}/models/current`);
    if (!res.ok) throw new Error(`Failed to get current model: ${res.status}`);
    return res.json();
  },

  /**
   * Predict single text sample.
   * Backend expects FormData fields: text, optional model_path, optional top_k
   */
  async predictText(text: string, modelPath?: string, topK: number = 3): Promise<any> {
    const form = new FormData();
    form.append("text", text);
    if (modelPath) form.append("model_path", modelPath);
    form.append("top_k", String(topK));

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
    });

    // predict may return 200 with ok:false payload, so parse JSON for details
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error(`Predict failed: ${res.status} ${JSON.stringify(json)}`);
    }
    return json;
  },

  /**
   * Predict batch samples.
   * This function prefers JSON body (backend accepts JSON for batch).
   * If you must use FormData (e.g., older fetch client), replace with a FormData implementation.
   */
  async predictBatch(samples: string[], modelPath?: string, topK: number = 3): Promise<any> {
    const body: any = { samples, top_k: topK };
    if (modelPath) body.model_path = modelPath;

    const res = await fetch(`${API_BASE}/predict/batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const json = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error(`Batch predict failed: ${res.status} ${JSON.stringify(json)}`);
    }
    return json;
  },
};

const API_URL = "https://modelforgeai.onrender.com";
