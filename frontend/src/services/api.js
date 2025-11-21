import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictSingle = async (transaction) => {
  const response = await api.post('/predict/single', { transaction });
  return response.data;
};

export const predictBatch = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/batch/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};


export const submitFeedback = async (feedbackData) => {
  const response = await api.post('/feedback/', feedbackData);
  return response.data;
};

export const getFeedbackStats = async () => {
  const response = await api.get('/feedback/stats');
  return response.data;
};

export const getTaxonomy = async () => {
  const response = await api.get('/taxonomy/categories');
  return response.data;
};

export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

export const getCategoryDistribution = async () => {
  const response = await api.get('/metrics/category-distribution');
  return response.data;
};

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
