import React from 'react';
import { View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useBaby } from '../../src/context/BabyContext';
import { useTheme } from '../../src/context/ThemeContext';
import BottomNavigation from '../../components/BottomNavigation';

export default function HistoryTabScreen() {
  const { cryEvents } = useBaby();
  const { colors, isDark } = useTheme();

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

        {/* History List */}
        {cryEvents && cryEvents.length > 0 ? (
          cryEvents.map((event, index) => (
            <View 
              key={index}
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
                    {event.type || 'Cry Detected'}
                  </Text>
                  <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>
                    {new Date(event.timestamp).toLocaleString()}
                  </Text>
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
        ) : (
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
              No recordings yet
            </Text>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
              Start recording to see your baby's cry history
            </Text>
          </View>
        )}
      </ScrollView>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
