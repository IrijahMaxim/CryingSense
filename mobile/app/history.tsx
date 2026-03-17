import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useBaby } from '../src/context/BabyContext';
import { useTheme } from '../src/context/ThemeContext';
import BottomNavigation from '../components/BottomNavigation';

export default function HistoryScreen() {
  const { cryEvents } = useBaby();
  const { colors, isDark } = useTheme();
  const [selectedFilter, setSelectedFilter] = useState<'today' | 'yesterday' | 'weekly'>('today');

  // Filter events by time period
  const getTodayEvents = () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return cryEvents.filter(event => new Date(event.timestamp) >= today);
  };

  const getYesterdayEvents = () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(0, 0, 0, 0);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return cryEvents.filter(event => {
      const eventDate = new Date(event.timestamp);
      return eventDate >= yesterday && eventDate < today;
    });
  };

  const getWeeklyEvents = () => {
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    return cryEvents.filter(event => new Date(event.timestamp) >= weekAgo);
  };

  const getFilteredEvents = () => {
    switch (selectedFilter) {
      case 'today':
        return getTodayEvents();
      case 'yesterday':
        return getYesterdayEvents();
      case 'weekly':
        return getWeeklyEvents();
      default:
        return getTodayEvents();
    }
  };

  const todayCount = getTodayEvents().length;
  const yesterdayCount = getYesterdayEvents().length;
  const weeklyCount = getWeeklyEvents().length;
  const filteredEvents = getFilteredEvents();

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <ScrollView style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text }}>History</Text>
          <TouchableOpacity onPress={() => router.push('/record')}>
            <Ionicons name="add-circle" size={32} color="#60A5FA" />
          </TouchableOpacity>
        </View>

        {/* Time Period Cards */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 }}>
          {/* Today Card */}
          <TouchableOpacity 
            style={{ 
              backgroundColor: selectedFilter === 'today' ? '#60A5FA' : (isDark ? '#2a2a2a' : 'white'), 
              padding: 16, 
              borderRadius: 16, 
              width: '31%', 
              alignItems: 'center',
              shadowColor: '#000', 
              shadowOffset: { width: 0, height: 2 }, 
              shadowOpacity: isDark ? 0.3 : 0.1, 
              shadowRadius: 4, 
              elevation: 3,
              borderWidth: selectedFilter === 'today' ? 2 : 0,
              borderColor: '#60A5FA'
            }}
            onPress={() => setSelectedFilter('today')}
          >
            <Ionicons name="today" size={24} color={selectedFilter === 'today' ? 'white' : '#60A5FA'} style={{ marginBottom: 8 }} />
            <Text style={{ fontSize: 14, fontWeight: '600', color: selectedFilter === 'today' ? 'white' : colors.text, marginBottom: 4 }}>Today</Text>
            <Text style={{ fontSize: 12, color: selectedFilter === 'today' ? 'rgba(255,255,255,0.8)' : (isDark ? '#999' : '#666') }}>{todayCount} recordings</Text>
          </TouchableOpacity>

          {/* Yesterday Card */}
          <TouchableOpacity 
            style={{ 
              backgroundColor: selectedFilter === 'yesterday' ? '#60A5FA' : (isDark ? '#2a2a2a' : 'white'), 
              padding: 16, 
              borderRadius: 16, 
              width: '31%', 
              alignItems: 'center',
              shadowColor: '#000', 
              shadowOffset: { width: 0, height: 2 }, 
              shadowOpacity: isDark ? 0.3 : 0.1, 
              shadowRadius: 4, 
              elevation: 3,
              borderWidth: selectedFilter === 'yesterday' ? 2 : 0,
              borderColor: '#60A5FA'
            }}
            onPress={() => setSelectedFilter('yesterday')}
          >
            <Ionicons name="calendar" size={24} color={selectedFilter === 'yesterday' ? 'white' : '#60A5FA'} style={{ marginBottom: 8 }} />
            <Text style={{ fontSize: 14, fontWeight: '600', color: selectedFilter === 'yesterday' ? 'white' : colors.text, marginBottom: 4 }}>Yesterday</Text>
            <Text style={{ fontSize: 12, color: selectedFilter === 'yesterday' ? 'rgba(255,255,255,0.8)' : (isDark ? '#999' : '#666') }}>{yesterdayCount} recordings</Text>
          </TouchableOpacity>

          {/* Weekly Card */}
          <TouchableOpacity 
            style={{ 
              backgroundColor: selectedFilter === 'weekly' ? '#60A5FA' : (isDark ? '#2a2a2a' : 'white'), 
              padding: 16, 
              borderRadius: 16, 
              width: '31%', 
              alignItems: 'center',
              shadowColor: '#000', 
              shadowOffset: { width: 0, height: 2 }, 
              shadowOpacity: isDark ? 0.3 : 0.1, 
              shadowRadius: 4, 
              elevation: 3,
              borderWidth: selectedFilter === 'weekly' ? 2 : 0,
              borderColor: '#60A5FA'
            }}
            onPress={() => setSelectedFilter('weekly')}
          >
            <Ionicons name="time" size={24} color={selectedFilter === 'weekly' ? 'white' : '#60A5FA'} style={{ marginBottom: 8 }} />
            <Text style={{ fontSize: 14, fontWeight: '600', color: selectedFilter === 'weekly' ? 'white' : colors.text, marginBottom: 4 }}>Weekly</Text>
            <Text style={{ fontSize: 12, color: selectedFilter === 'weekly' ? 'rgba(255,255,255,0.8)' : (isDark ? '#999' : '#666') }}>{weeklyCount} recordings</Text>
          </TouchableOpacity>
        </View>

        {/* Filtered Recordings */}
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: colors.text, marginBottom: 16 }}>
          {selectedFilter === 'today' ? 'Today\'s Recordings' : selectedFilter === 'yesterday' ? 'Yesterday\'s Recordings' : 'Weekly Recordings'}
        </Text>
        
        {filteredEvents.length === 0 ? (
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 32, 
            borderRadius: 12, 
            alignItems: 'center', 
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Ionicons name="time-outline" size={48} color={isDark ? '#666' : '#ccc'} style={{ marginBottom: 16 }} />
            <Text style={{ fontSize: 16, color: colors.text, textAlign: 'center', marginBottom: 8 }}>
              No recordings {selectedFilter === 'today' ? 'today' : selectedFilter === 'yesterday' ? 'yesterday' : 'this week'}
            </Text>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
              Start recording to see your baby's cry history
            </Text>
          </View>
        ) : (
          filteredEvents.map((event) => (
            <View 
              key={event.id} 
              style={{ 
                backgroundColor: isDark ? '#2a2a2a' : 'white', 
                padding: 16, 
                borderRadius: 12, 
                marginBottom: 12, 
                shadowColor: '#000', 
                shadowOffset: { width: 0, height: 2 }, 
                shadowOpacity: isDark ? 0.3 : 0.1, 
                shadowRadius: 4, 
                elevation: 3 
              }}
            >
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                <View>
                  <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 4 }}>
                    {event.need}
                  </Text>
                  <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>
                    {new Date(event.timestamp).toLocaleString()}
                  </Text>
                  <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 4 }}>
                    <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666', marginRight: 8 }}>
                      Duration: {event.duration}s
                    </Text>
                    <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>
                      Confidence: {event.confidence}%
                    </Text>
                  </View>
                </View>
                <View style={{ 
                  backgroundColor: '#4CAF50', 
                  paddingHorizontal: 8, 
                  paddingVertical: 4, 
                  borderRadius: 12 
                }}>
                  <Text style={{ fontSize: 12, color: 'white', fontWeight: '600' }}>Analyzed</Text>
                </View>
              </View>
            </View>
          ))
        )}
      </ScrollView>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
