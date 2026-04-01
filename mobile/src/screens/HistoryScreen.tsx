import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, FlatList, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useBaby } from '../context/BabyContext';
import { useTheme } from '../context/ThemeContext';
import styles from '../styles/AppStyles';
import GradientBackground from '../components/GradientBackground';
import FirebaseService, { FilteredCryEvent } from '../services/FirebaseService';

type FilterOption = 'All' | 'Today' | 'Yesterday' | 'Week';

export default function HistoryAndInsightsScreen({ navigation }: any) {
  const { cryEvents, isLoading, refreshCryEvents } = useBaby();
  const { isDark } = useTheme();
  const [filter, setFilter] = useState<FilterOption>('All');
  const [firebaseEvents, setFirebaseEvents] = useState<FilteredCryEvent[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const firebaseService = FirebaseService.getInstance();
  
  // Load Firebase events on component mount
  useEffect(() => {
    loadFirebaseEvents();
  }, []);
  
  // Load events from Firebase with filter
  const loadFirebaseEvents = async () => {
    setIsRefreshing(true);
    try {
      const events = await firebaseService.fetchCryingHistoryWithFilter(filter);
      setFirebaseEvents(events);
    } catch (error) {
      console.error('Error loading Firebase events:', error);
    } finally {
      setIsRefreshing(false);
    }
  };
  
  // Update events when filter changes
  useEffect(() => {
    loadFirebaseEvents();
  }, [filter]);

  // Use Firebase events instead of mock data
  const filteredEvents = firebaseEvents.map(event => ({
    id: event.id,
    category: event.category,
    timestamp: event.timestamp,
    need: event.category,
    confidence: Math.round(event.confidence * 100),
    duration: event.duration
  }));

  // Render each recording
  const renderRecording = ({ item }: { item: any }) => (
    <View style={{
      backgroundColor: isDark ? '#2a2a2a' : 'white',
      padding: 12,
      borderRadius: 12,
      marginBottom: 15,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: 4,
      elevation: 4,
      width: '95%', alignSelf: 'center'
    }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <View style={{ 
                width: 24, 
                height: 24, 
                borderRadius: 12, 
                backgroundColor: isDark ? '#60A5FA' : 'rgba(96, 165, 250, 0.8)', 
                justifyContent: 'center', 
                alignItems: 'center', 
                marginRight: 8 
              }}>
                <Ionicons name="volume-medium" size={16} color="white" />
              </View>
            <View>
              <Text style={{ fontSize: 13, fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333', marginBottom: 4 }}>
                {new Date(item.timestamp).toLocaleDateString()}
              </Text>
              <Text style={{ 
                fontSize: 12, 
                color: '#60A5FA' 
              }}>
                {new Date(item.timestamp).toLocaleTimeString()}
              </Text>
            </View>
          </View>
        <View style={{ alignItems: 'flex-end' }}>
          <Text style={{
            fontSize: 11,
            fontWeight: 'normal',
            color: item.category === 'Hungry' ? '#FF6347' : 
                   item.category === 'Sleepy' ? '#4CAF50' : 
                   item.category === 'Discomfort' ? '#FF9800' : '#60A5FA'
          }}>
            {item.category}
          </Text>
          <Text style={{ 
            fontSize: 11, 
            color: '#60A5FA' 
          }}>
            {item.confidence ? `${(item.confidence * 100).toFixed(0)}%` : 'N/A'}
          </Text>
        </View>
      </View>
    </View>
  );

  return (
      <View style={[styles.container, { backgroundColor: 'transparent', flex: 1 }]}>
        {/* -------- Header -------- */}
        <View style={styles.header}>
          <View style={{ width: 28 }} /> {/* placeholder for alignment */}
          <View style={{ width: 28 }} /> {/* placeholder for alignment */}
        </View>

      {/* -------- Filter Options -------- */}
      <View style={{ 
        flexDirection: 'row', 
        marginBottom: 12,
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}>
        {(['All', 'Today', 'Yesterday', 'Week'] as FilterOption[]).map((option) => (
          <TouchableOpacity
            key={option}
            style={{
              paddingVertical: 6,
              paddingHorizontal: 12,
              borderRadius: 12,
              marginRight: 8,
              backgroundColor: filter === option ? '#60A5FA' : (isDark ? '#2a2a2a' : '#f0f0f0'),
            }}
            onPress={() => setFilter(option)}
          >
            <Text style={{
              fontSize: 11,
              fontWeight: 'normal',
              color: filter === option ? 'white' : (isDark ? '#ECEDEE' : '#333'),
            }}>
              {option}
            </Text>
          </TouchableOpacity>
        ))}
        
        {/* Refresh Button */}
        <TouchableOpacity
          style={{
            paddingVertical: 6,
            paddingHorizontal: 12,
            borderRadius: 12,
            backgroundColor: '#4CAF50',
          }}
          onPress={loadFirebaseEvents}
          disabled={isRefreshing}
        >
          {isRefreshing ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Ionicons name="refresh" size={16} color="white" />
          )}
        </TouchableOpacity>
      </View>

      {/* -------- Loading State -------- */}
      {(isLoading || isRefreshing) && filteredEvents.length === 0 && (
        <View style={{ 
          flex: 1, 
          justifyContent: 'center', 
          alignItems: 'center',
          marginTop: 100
        }}>
          <ActivityIndicator size="large" color="#60A5FA" />
          <Text style={{ 
            fontSize: 14, 
            color: isDark ? '#ECEDEE' : '#333', 
            marginTop: 16,
            textAlign: 'center'
          }}>
            Loading cry detections from Firebase...
          </Text>
        </View>
      )}

      {/* -------- Recording List -------- */}
      <FlatList
        data={filteredEvents}
        keyExtractor={(item) => item.id}
        renderItem={renderRecording}
        contentContainerStyle={{ paddingBottom: 80 }}
        showsVerticalScrollIndicator={true}
        style={{ flex: 1 }}
        ListEmptyComponent={!isLoading && !isRefreshing ? (
          <View style={{ 
            flex: 1, 
            justifyContent: 'center', 
            alignItems: 'center',
            marginTop: 100,
            paddingHorizontal: 20
          }}>
            <Ionicons name="time-outline" size={48} color={isDark ? '#666' : '#ccc'} style={{ marginBottom: 16 }} />
            <Text style={{ 
              fontSize: 16, 
              color: isDark ? '#ECEDEE' : '#333', 
              textAlign: 'center',
              marginBottom: 8
            }}>
              No crying detections found
            </Text>
            <Text style={{ 
              fontSize: 12, 
              color: isDark ? '#999' : '#666', 
              textAlign: 'center'
            }}>
              ESP32 device detections will appear here once connected to Firebase
            </Text>
          </View>
        ) : null}
      />
    </View>
  );
}
