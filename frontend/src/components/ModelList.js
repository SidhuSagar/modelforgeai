import React, { useEffect, useState } from "react";
import { listModels } from "../api";

const ModelList = ({ onSelect }) => {
  const [models, setModels] = useState([]);

  useEffect(() => {
    const fetchModels = async () => {
      const data = await listModels();
      setModels(data);
    };
    fetchModels();
  }, []);

  return (
    <div>
      <h2>Available Models</h2>
      <ul>
        {models.map((m) => (
          <li key={m}>
            {m} <button onClick={() => onSelect(m)}>Select</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ModelList;
