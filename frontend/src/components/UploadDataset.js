import React, { useState } from "react";
import { uploadDataset } from "../api";

const UploadDataset = () => {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [modelType, setModelType] = useState("random_forest");
  const [status, setStatus] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) return alert("Select file and target column");
    setStatus("Training...");
    try {
      const res = await uploadDataset(file, target, modelType);
      setStatus(res.message);
    } catch (err) {
      setStatus("Error: " + err.message);
    }
  };

  return (
    <div>
      <h2>Upload Dataset for Training</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <input type="text" placeholder="Target Column" value={target} onChange={(e) => setTarget(e.target.value)} />
        <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
          <option value="random_forest">Random Forest</option>
          <option value="logistic_regression">Logistic Regression</option>
        </select>
        <button type="submit">Train Model</button>
      </form>
      <p>{status}</p>
    </div>
  );
};

export default UploadDataset;
