import React, { useState } from "react";
import { predict } from "../api";

const PredictForm = ({ model }) => {
  const [inputData, setInputData] = useState("");
  const [result, setResult] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!model || !inputData) return alert("Select model and enter input");
    const data = JSON.parse(inputData);
    const res = await predict(model, data);
    setResult(JSON.stringify(res));
  };

  return (
    <div>
      <h2>Talk to Model: {model || "Select a model"}</h2>
      <form onSubmit={handleSubmit}>
        <textarea value={inputData} onChange={(e) => setInputData(e.target.value)} placeholder='Enter JSON input' rows={5} cols={50} />
        <button type="submit">Predict</button>
      </form>
      <pre>{result}</pre>
    </div>
  );
};

export default PredictForm;
