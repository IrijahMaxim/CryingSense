import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useBaby } from '../../src/context/BabyContext';
import { useTheme } from '../../src/context/ThemeContext';
import BottomNavigation from '../../components/BottomNavigation';

export default function HomeScreen() {
  const { profile, getTodayEvents, getTopNeed, getTotalRecordings } = useBaby();
  const { colors, isDark } = useTheme();

  const todayEvents = getTodayEvents();
  const topNeed = getTopNeed();
  const totalRecordings = getTotalRecordings();

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <ScrollView style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text }}>CryingSense</Text>
          <TouchableOpacity onPress={() => router.push('/settings')}>
            <Image 
              source={require('../../assets/images/logo.png')} 
              style={{ width: 32, height: 32 }}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

        {/* Profile Section */}
        <TouchableOpacity 
          style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 12, marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3 }}
          onPress={() => router.push('/profile-details')}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            {profile?.photo ? (
              <Image
                source={{ uri: profile.photo }}
                style={{ width: 60, height: 60, borderRadius: 30, marginRight: 16 }}
              />
            ) : (
              <Image
                source={require('../../assets/baby_placeholder.png')}
                style={{ width: 60, height: 60, borderRadius: 30, marginRight: 16 }}
              />
            )}
            <View>
              <Text style={{ fontSize: 18, fontWeight: '600', color: colors.text, marginBottom: 4 }}>{profile?.name || 'Baby'}</Text>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>{profile?.ageMonths || 0} months old</Text>
            </View>
          </View>
        </TouchableOpacity>

        {/* Today's Insights Section */}
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: colors.text, marginBottom: 16 }}>Today's Insights</Text>
        
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 }}>
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 20, 
            borderRadius: 12, 
            width: '48%', 
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 8 }}>Total Recordings</Text>
            <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text }}>{totalRecordings}</Text>
          </View>
          
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 20, 
            borderRadius: 12, 
            width: '48%', 
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 8 }}>Top Need</Text>
            <Text style={{ fontSize: 18, fontWeight: '600', color: '#60A5FA' }}>{topNeed}</Text>
          </View>
        </View>

        {/* Recent Activity Section */}
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: colors.text, marginBottom: 16 }}>Recent Activity</Text>
        
        <View style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 20, 
          borderRadius: 12, 
          marginBottom: 20, 
          shadowColor: '#000', 
          shadowOffset: { width: 0, height: 2 }, 
          shadowOpacity: isDark ? 0.3 : 0.1, 
          shadowRadius: 4, 
          elevation: 3 
        }}>
          {todayEvents.length > 0 ? (
            todayEvents.slice(0, 3).map((event, index) => (
              <View 
                key={event.id}
                style={{ 
                  backgroundColor: isDark ? '#333' : '#f5f5f5', 
                  padding: 16, 
                  borderRadius: 8, 
                  marginBottom: index < todayEvents.slice(0, 3).length - 1 ? 12 : 0, 
                  borderWidth: 1, 
                  borderColor: isDark ? '#555' : '#e0e0e0'
                }}
              >
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                  <View>
                    <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 4 }}>{event.need}</Text>
                    <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>
                      {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
            <View style={{ alignItems: 'center', padding: 20 }}>
              <Ionicons name="time-outline" size={32} color={isDark ? '#666' : '#ccc'} style={{ marginBottom: 8 }} />
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
                No recordings today
              </Text>
            </View>
          )}
        </View>
      </ScrollView>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
