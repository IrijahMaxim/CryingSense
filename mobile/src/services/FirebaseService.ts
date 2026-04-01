// FirebaseService - FRONTEND ONLY
// 
// This service provides Firebase integration for CryingSense mobile app
// No backend dependencies - pure Firebase Firestore
// 
// ============================================================================

import { initializeApp, getApps, getApp } from 'firebase/app';
import { getFirestore, collection, getDocs, query, orderBy, where, Timestamp, onSnapshot, doc, getDoc, addDoc, updateDoc, deleteDoc } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Firebase configuration - your actual project config
const firebaseConfig = {
  apiKey: "AIzaSyBhKZ7r8xqL9pV2nK5sT7mJ6fQ3rW8kX",
  authDomain: "cryingsense.firebaseapp.com",
  projectId: "cryingsense",
  storageBucket: "cryingsense.firebasestorage.app",
  messagingSenderId: "297101442271",
  appId: "1:297101442271:web:abcdef123456"
};

// Initialize Firebase
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const db = getFirestore(app);

export interface CryDetection {
  id: string;
  timestamp: Timestamp;
  soundLevel: number;
  classification: 'quiet' | 'normal' | 'crying';
  confidence: number;
  duration: number;
  deviceId?: string;
  deviceName?: string;
  location?: string;
  babyProfile?: string;
  notes?: string;
}

export interface FilteredCryEvent {
  id: string;
  timestamp: Date;
  classification: string;
  confidence: number;
  duration: number;
  deviceId?: string;
  deviceName?: string;
  soundLevel: number;
  category: string;
}

class FirebaseService {
  private static instance: FirebaseService;
  private unsubscribeRealtime: (() => void) | null = null;

  private constructor() {}

  static getInstance(): FirebaseService {
    if (!FirebaseService.instance) {
      FirebaseService.instance = new FirebaseService();
    }
    return FirebaseService.instance;
  }

  // Fetch all cry detections from Firestore
  async fetchCryDetections(): Promise<CryDetection[]> {
    try {
      const cryCollection = collection(db, 'cryDetections');
      const q = query(cryCollection, orderBy('timestamp', 'desc'));
      const querySnapshot = await getDocs(q);
      
      const detections: CryDetection[] = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        detections.push({
          id: doc.id,
          timestamp: data.timestamp,
          soundLevel: data.soundLevel || 0,
          classification: data.classification || 'normal',
          confidence: data.confidence || 0,
          duration: data.duration || 0,
          deviceId: data.deviceId,
          deviceName: data.deviceName,
          location: data.location,
          babyProfile: data.babyProfile,
          notes: data.notes
        });
      });
      
      return detections;
    } catch (error) {
      console.error('Error fetching cry detections:', error);
      return [];
    }
  }

  // Filter out quiet and normal detections, only return crying
  filterCryingDetections(detections: CryDetection[]): FilteredCryEvent[] {
    return detections
      .filter(detection => detection.classification === 'crying')
      .map(detection => ({
        id: detection.id,
        timestamp: detection.timestamp.toDate(),
        classification: detection.classification,
        confidence: detection.confidence,
        duration: detection.duration,
        deviceId: detection.deviceId,
        deviceName: detection.deviceName,
        soundLevel: detection.soundLevel,
        category: this.getCategoryFromConfidence(detection.confidence)
      }));
  }

  // Get category based on confidence level
  private getCategoryFromConfidence(confidence: number): string {
    if (confidence >= 0.8) return 'Hungry';
    if (confidence >= 0.6) return 'Sleepy';
    if (confidence >= 0.4) return 'Discomfort';
    return 'Attention';
  }

  // Fetch filtered crying events for history
  async fetchCryingHistory(): Promise<FilteredCryEvent[]> {
    const detections = await this.fetchCryDetections();
    return this.filterCryingDetections(detections);
  }

  // Fetch crying events with date filtering
  async fetchCryingHistoryWithFilter(filter: 'All' | 'Today' | 'Yesterday' | 'Week'): Promise<FilteredCryEvent[]> {
    const detections = await this.fetchCryingHistory();
    const now = new Date();
    
    return detections.filter(event => {
      const eventDate = event.timestamp;
      
      switch (filter) {
        case 'Today':
          return eventDate.toDateString() === now.toDateString();
        case 'Yesterday':
          const yesterday = new Date(now);
          yesterday.setDate(now.getDate() - 1);
          return eventDate.toDateString() === yesterday.toDateString();
        case 'Week':
          const weekAgo = new Date(now);
          weekAgo.setDate(now.getDate() - 7);
          return eventDate >= weekAgo;
        case 'All':
        default:
          return true;
      }
    });
  }

  // Set up real-time listener for crying detections
  setupRealtimeListener(callback: (events: FilteredCryEvent[]) => void): void {
    // Clean up existing listener
    if (this.unsubscribeRealtime) {
      this.unsubscribeRealtime();
    }
    
    const cryCollection = collection(db, 'cryDetections');
    const q = query(cryCollection, orderBy('timestamp', 'desc'));
    
    this.unsubscribeRealtime = onSnapshot(q, (querySnapshot) => {
      const detections: CryDetection[] = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        detections.push({
          id: doc.id,
          timestamp: data.timestamp,
          soundLevel: data.soundLevel || 0,
          classification: data.classification || 'normal',
          confidence: data.confidence || 0,
          duration: data.duration || 0,
          deviceId: data.deviceId,
          deviceName: data.deviceName,
          location: data.location,
          babyProfile: data.babyProfile,
          notes: data.notes
        });
      });
      
      const filteredEvents = this.filterCryingDetections(detections);
      callback(filteredEvents);
    });
  }

  // Stop real-time listener
  stopRealtimeListener(): void {
    if (this.unsubscribeRealtime) {
      this.unsubscribeRealtime();
      this.unsubscribeRealtime = null;
    }
  }

  // Get specific detection by ID
  async getDetectionById(id: string): Promise<CryDetection | null> {
    try {
      const docRef = doc(db, 'cryDetections', id);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        const data = docSnap.data();
        return {
          id: docSnap.id,
          timestamp: data.timestamp,
          soundLevel: data.soundLevel || 0,
          classification: data.classification || 'normal',
          confidence: data.confidence || 0,
          duration: data.duration || 0,
          deviceId: data.deviceId,
          deviceName: data.deviceName,
          location: data.location,
          babyProfile: data.babyProfile,
          notes: data.notes
        };
      }
      return null;
    } catch (error) {
      console.error('Error getting detection:', error);
      return null;
    }
  }

  // Cache data locally for offline use
  async cacheData(events: FilteredCryEvent[]): Promise<void> {
    try {
      await AsyncStorage.setItem('cry-history-cache', JSON.stringify(events));
      await AsyncStorage.setItem('cry-history-cache-time', new Date().toISOString());
    } catch (error) {
      console.error('Error caching data:', error);
    }
  }

  // Get cached data
  async getCachedData(): Promise<FilteredCryEvent[] | null> {
    try {
      const cached = await AsyncStorage.getItem('cry-history-cache');
      if (cached) {
        const events = JSON.parse(cached);
        return events.map((event: any) => ({
          ...event,
          timestamp: new Date(event.timestamp)
        }));
      }
      return null;
    } catch (error) {
      console.error('Error getting cached data:', error);
      return null;
    }
  }

  // Check if cache is fresh (less than 5 minutes old)
  async isCacheFresh(): Promise<boolean> {
    try {
      const cacheTime = await AsyncStorage.getItem('cry-history-cache-time');
      if (cacheTime) {
        const cacheDate = new Date(cacheTime);
        const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
        return cacheDate > fiveMinutesAgo;
      }
      return false;
    } catch (error) {
      console.error('Error checking cache freshness:', error);
      return false;
    }
  }

  // Add new cry detection to Firebase
  async addCryDetection(cryData: Omit<CryDetection, 'id'>): Promise<string | null> {
    try {
      const docRef = await addDoc(collection(db, 'cryDetections'), {
        ...cryData,
        timestamp: Timestamp.now()
      });
      return docRef.id;
    } catch (error) {
      console.error('Error adding cry detection:', error);
      return null;
    }
  }

  // Update cry detection
  async updateCryDetection(id: string, data: Partial<CryDetection>): Promise<boolean> {
    try {
      await updateDoc(doc(db, 'cryDetections', id), data);
      return true;
    } catch (error) {
      console.error('Error updating cry detection:', error);
      return false;
    }
  }

  // Delete cry detection
  async deleteCryDetection(id: string): Promise<boolean> {
    try {
      await deleteDoc(doc(db, 'cryDetections', id));
      return true;
    } catch (error) {
      console.error('Error deleting cry detection:', error);
      return false;
    }
  }

  // Get Firebase config for user to update
  getFirebaseConfig(): typeof firebaseConfig {
    return firebaseConfig;
  }

  // Update Firebase config (for user setup)
  updateFirebaseConfig(config: Partial<typeof firebaseConfig>): void {
    Object.assign(firebaseConfig, config);
  }
}

export default FirebaseService;
