import axios from "axios";

const BASE_URL = "http://127.0.0.1:8080/api";

export const listModels = async () => {
  const res = await axios.get(`${BASE_URL}/models`);
  return res.data.models;
};

export const predict = async (modelName, inputData) => {
  const res = await axios.post(`${BASE_URL}/predict`, { model: modelName, data: inputData });
  return res.data;
};

export const uploadDataset = async (file, targetColumn, modelType) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target_column", targetColumn);
  formData.append("model_type", modelType);

  const res = await axios.post(`${BASE_URL}/train`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};
