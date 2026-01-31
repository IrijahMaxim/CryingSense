import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Default backend URL
const DEFAULT_BASE_URL = 'http://localhost:3000/api';

// Get base URL from storage or use default
const getBaseUrl = async () => {
  try {
    const url = await AsyncStorage.getItem('backendUrl');
    return url || DEFAULT_BASE_URL;
  } catch (error) {
    console.error('Error getting base URL:', error);
    return DEFAULT_BASE_URL;
  }
};

// Fetch latest classifications
export const fetchLatestClassifications = async (limit = 10) => {
  try {
    const baseUrl = await getBaseUrl();
    const response = await axios.get(`${baseUrl}/classifications`, {
      params: { limit, page: 1 },
      timeout: 5000,
    });
    return response.data.data || [];
  } catch (error) {
    console.error('Error fetching classifications:', error);
    // Return mock data for development/testing
    return mockClassifications.slice(0, limit);
  }
};

// Fetch classification by ID
export const fetchClassificationById = async (id) => {
  try {
    const baseUrl = await getBaseUrl();
    const response = await axios.get(`${baseUrl}/classifications/${id}`, {
      timeout: 5000,
    });
    return response.data.data;
  } catch (error) {
    console.error('Error fetching classification:', error);
    return null;
  }
};

// Fetch statistics
export const fetchStatistics = async (days = 7) => {
  try {
    const baseUrl = await getBaseUrl();
    const response = await axios.get(`${baseUrl}/classifications/stats`, {
      params: { days },
      timeout: 5000,
    });
    return response.data.data;
  } catch (error) {
    console.error('Error fetching statistics:', error);
    return null;
  }
};

// Mock data for development
const mockClassifications = [
  {
    _id: '1',
    deviceId: 'ESP32_001',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    predictedClass: 'hunger',
    confidence: 0.87,
    probabilities: {
      belly_pain: 0.05,
      burp: 0.03,
      discomfort: 0.02,
      hunger: 0.87,
      tired: 0.03,
    },
    inferenceTimeMs: 125.4,
  },
  {
    _id: '2',
    deviceId: 'ESP32_001',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    predictedClass: 'tired',
    confidence: 0.92,
    probabilities: {
      belly_pain: 0.02,
      burp: 0.01,
      discomfort: 0.03,
      hunger: 0.02,
      tired: 0.92,
    },
    inferenceTimeMs: 118.2,
  },
  {
    _id: '3',
    deviceId: 'ESP32_001',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    predictedClass: 'discomfort',
    confidence: 0.78,
    probabilities: {
      belly_pain: 0.10,
      burp: 0.05,
      discomfort: 0.78,
      hunger: 0.04,
      tired: 0.03,
    },
    inferenceTimeMs: 132.6,
  },
];

export default {
  fetchLatestClassifications,
  fetchClassificationById,
  fetchStatistics,
};
