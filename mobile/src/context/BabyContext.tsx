// context/BabyContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import FirebaseService, { FilteredCryEvent } from '../services/FirebaseService';

type BabyProfile = {
  name: string;
  ageMonths: number;
  photo?: string | null; // URL or local asset path
};

export type CryEvent = {
  id: string;
  category: string;
  timestamp: Date;
  need: string; // Detected need (Hungry, Sleepy, Uncomfortable, etc.)
  confidence: number; // Confidence score 0-100
  duration: number; // Recording duration in seconds
};

type BabyContextType = {
  profile: BabyProfile;
  setProfile: (profile: BabyProfile) => void;
  cryEvents: CryEvent[];
  addCryEvent: (event: CryEvent) => void;
  getTodayEvents: () => CryEvent[];
  getTopNeed: () => string;
  getTotalRecordings: () => number;
  refreshCryEvents: () => Promise<void>;
  isLoading: boolean;
};

const BabyContext = createContext<BabyContextType | undefined>(undefined);

// Sample test data
const sampleCryEvents: CryEvent[] = [
  {
    id: '1',
    category: 'Hungry',
    timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    need: 'Hungry',
    confidence: 92,
    duration: 15
  },
  {
    id: '2',
    category: 'Sleepy',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
    need: 'Sleepy',
    confidence: 88,
    duration: 22
  },
  {
    id: '3',
    category: 'Uncomfortable',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4), // 4 hours ago
    need: 'Uncomfortable',
    confidence: 75,
    duration: 18
  },
  {
    id: '4',
    category: 'Hungry',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6), // 6 hours ago
    need: 'Hungry',
    confidence: 95,
    duration: 25
  },
  {
    id: '5',
    category: 'Sleepy',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 8), // 8 hours ago
    need: 'Sleepy',
    confidence: 91,
    duration: 30
  },
  {
    id: '6',
    category: 'Attention',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 12), // 12 hours ago
    need: 'Attention',
    confidence: 83,
    duration: 12
  },
  {
    id: '7',
    category: 'Hungry',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 15), // 15 hours ago
    need: 'Hungry',
    confidence: 89,
    duration: 20
  },
  {
    id: '8',
    category: 'Uncomfortable',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 18), // 18 hours ago
    need: 'Uncomfortable',
    confidence: 78,
    duration: 16
  },
  {
    id: '9',
    category: 'Sleepy',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 22), // 22 hours ago
    need: 'Sleepy',
    confidence: 94,
    duration: 28
  },
  {
    id: '10',
    category: 'Hungry',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 25), // 25 hours ago
    need: 'Hungry',
    confidence: 87,
    duration: 19
  }
];

export const BabyProvider = ({ children }: { children: ReactNode }) => {
  const [profile, setProfile] = useState<BabyProfile>({ name: 'Baby', ageMonths: 6 });
  const [cryEvents, setCryEvents] = useState<CryEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const firebaseService = FirebaseService.getInstance();
  
  // Load cry events from Firebase on component mount
  useEffect(() => {
    loadCryEvents();
    
    // Set up real-time listener
    firebaseService.setupRealtimeListener((events) => {
      const formattedEvents = events.map(event => ({
        id: event.id,
        category: event.category,
        timestamp: event.timestamp,
        need: event.category,
        confidence: Math.round(event.confidence * 100),
        duration: event.duration
      }));
      setCryEvents(formattedEvents);
    });
    
    return () => {
      firebaseService.stopRealtimeListener();
    };
  }, []);
  
  const loadCryEvents = async () => {
    setIsLoading(true);
    try {
      const events = await firebaseService.fetchCryingHistory();
      const formattedEvents = events.map(event => ({
        id: event.id,
        category: event.category,
        timestamp: event.timestamp,
        need: event.category,
        confidence: Math.round(event.confidence * 100),
        duration: event.duration
      }));
      setCryEvents(formattedEvents);
    } catch (error) {
      console.error('Error loading cry events:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const refreshCryEvents = async () => {
    await loadCryEvents();
  };

  const addCryEvent = (event: CryEvent) => {
    setCryEvents([...cryEvents, event]);
  };

  const getTodayEvents = () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return cryEvents.filter(event => new Date(event.timestamp) >= today);
  };

  const getTopNeed = () => {
    if (cryEvents.length === 0) return 'No data';
    
    const needCounts = cryEvents.reduce((acc, event) => {
      acc[event.need] = (acc[event.need] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return Object.entries(needCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0];
  };

  const getTotalRecordings = () => {
    return cryEvents.length;
  };

  return (
    <BabyContext.Provider value={{ 
      profile, 
      setProfile, 
      cryEvents, 
      addCryEvent, 
      getTodayEvents, 
      getTopNeed, 
      getTotalRecordings,
      refreshCryEvents,
      isLoading
    }}>
      {children}
    </BabyContext.Provider>
  );
};

export const useBaby = () => {
  const context = useContext(BabyContext);
  if (!context) throw new Error('useBaby must be used within BabyProvider');
  return context;
};
