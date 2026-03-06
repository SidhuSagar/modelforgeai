import React, { useState } from "react";
import UploadDataset from "./components/UploadDataset";
import ModelList from "./components/ModelList";
import PredictForm from "./components/PredictForm";

function App() {
  const [selectedModel, setSelectedModel] = useState("");

  return (
    <div style={{ padding: "20px" }}>
      <h1>ModelForge AI Interface</h1>
      <UploadDataset />
      <hr />
      <ModelList onSelect={setSelectedModel} />
      <hr />
      <PredictForm model={selectedModel} />
    </div>
  );
}

export default App;
