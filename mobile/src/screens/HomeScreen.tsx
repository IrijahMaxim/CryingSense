// screens/HomeScreen.tsx
import React, { useState } from 'react';
import { View, Text, ScrollView, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useBaby } from '../context/BabyContext';
import { useTheme } from '../context/ThemeContext';
import RecordScreen from './RecordScreen';
import HistoryAndInsightsScreen from './HistoryScreen';
import SettingsScreen from './SettingsScreen';
import EditBabyProfileScreen from './EditBabyProfileScreen';
import AnalysisResultScreen from './AnalysisResultScreen';
import GradientBackground from '../components/GradientBackground';

export default function HomeScreen() {
  const { profile, cryEvents } = useBaby();
  const { colors, isDark } = useTheme();
  const [currentScreen, setCurrentScreen] = useState('home');

  const today = new Date();
  const todayEvents = cryEvents.filter((e) => e.timestamp.toDateString() === today.toDateString());

  const topNeed = todayEvents.reduce<Record<string, number>>((acc, e) => {
    acc[e.category] = (acc[e.category] || 0) + 1;
    return acc;
  }, {});
  const topCategory =
    Object.keys(topNeed).reduce((a, b) => (topNeed[a] > topNeed[b] ? a : b), '') || 'None';

  const renderScreen = () => {
    switch (currentScreen) {
      case 'record':
        return <RecordScreenWrapper onBack={() => setCurrentScreen('home')} onNavigateToAnalysis={() => setCurrentScreen('analysis-result')} />;
      case 'history':
        return <HistoryScreenWrapper onBack={() => setCurrentScreen('home')} />;
      case 'settings':
        return <SettingsScreenWrapper onBack={() => setCurrentScreen('home')} onNavigateToEditProfile={() => setCurrentScreen('edit-profile')} />;
      case 'edit-profile':
        return <EditProfileScreenWrapper onBack={() => setCurrentScreen('home')} />;
      case 'analysis-result':
        return <AnalysisScreenWrapper onBack={() => setCurrentScreen('home')} onNavigateToRecord={() => setCurrentScreen('record')} />;
      default:
        return null;
    }
  };

  const renderContent = () => {
    if (currentScreen !== 'home') {
      return renderScreen();
    }

    return (
      <GradientBackground>
        <View style={{ flex: 1, paddingHorizontal: 16, paddingTop: 40 }}>
          {/* Logo */}
          <View style={{ alignItems: 'center', marginBottom: 24 }}>
            <Image
              source={require('../../assets/images/logo.png')}
              style={{ width: 48, height: 48 }}
              resizeMode="contain"
            />
          </View>

          {/* Baby Info Card */}
          <TouchableOpacity 
            style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 12, borderRadius: 16, marginBottom: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3, width: '95%', alignSelf: 'center' }}
            onPress={() => setCurrentScreen('edit-profile')}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Image
                source={require('../../assets/baby_placeholder.png')}
                style={{ width: 60, height: 60, borderRadius: 30, marginRight: 14 }}
              />
              <View>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 3 }}>{profile.name}</Text>
                <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>{profile.ageMonths} months old</Text>
              </View>
            </View>
          </TouchableOpacity>

          {/* Today's Insights */}
          <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 16, textAlign: 'center' }}>Today's Insights</Text>
          
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 16, width: '95%', alignSelf: 'center' }}>
            <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 14, borderRadius: 16, width: '48%', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3 }}>
              <Text style={{ fontSize: 11, color: isDark ? '#999' : '#666', marginBottom: 6 }}>Total Recordings</Text>
              <Text style={{ fontSize: 18, fontWeight: 'normal', color: colors.text }}>{cryEvents.length}</Text>
            </View>
            
            <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 14, borderRadius: 16, width: '48%', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3 }}>
              <Text style={{ fontSize: 11, color: isDark ? '#999' : '#666', marginBottom: 6 }}>Top Need</Text>
              <Text style={{ fontSize: 13, fontWeight: 'normal', color: '#60A5FA' }}>{topCategory}</Text>
            </View>
          </View>

          {/* Recent Activity */}
          <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 16, textAlign: 'center' }}>Recent Activity</Text>
          
          {todayEvents.length > 0 ? (
            todayEvents.slice(0, 3).map((event) => (
              <View key={event.id} style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 12, borderRadius: 12, marginBottom: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3, width: '95%', alignSelf: 'center' }}>
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
                      <Text style={{ fontSize: 16, fontWeight: 'normal', color: colors.text, marginBottom: 4 }}>{event.category}</Text>
                      <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>
                        {event.timestamp.toLocaleTimeString()}
                      </Text>
                    </View>
                  </View>
                  <View style={{ backgroundColor: '#4CAF50', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 12 }}>
                    <Text style={{ fontSize: 11, color: 'white', fontWeight: '600' }}>Analyzed</Text>
                  </View>
                </View>
              </View>
            ))
          ) : (
            <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 18, borderRadius: 16, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 4, elevation: 3, marginBottom: 15, width: '95%', alignSelf: 'center' }}>
              <Ionicons name="time-outline" size={32} color={isDark ? '#666' : '#ccc'} style={{ marginBottom: 8 }} />
              <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
                No recordings today
              </Text>
            </View>
          )}
        </View>
      </GradientBackground>
    );
  };

  return (
    <GradientBackground>
      {renderContent()}
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          {/* Home */}
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={() => setCurrentScreen('home')}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={currentScreen === 'home' ? '#60A5FA' : isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>

          {/* Recording */}
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={() => setCurrentScreen('record')}
          >
            <Ionicons 
              name="mic" 
              size={24} 
              color={currentScreen === 'record' ? '#60A5FA' : isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>

          {/* History */}
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={() => setCurrentScreen('history')}
          >
            <Ionicons 
              name="time" 
              size={24} 
              color={currentScreen === 'history' ? '#60A5FA' : isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>

          {/* Settings */}
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={() => setCurrentScreen('settings')}
          >
            <Ionicons 
              name="settings" 
              size={24} 
              color={currentScreen === 'settings' ? '#60A5FA' : isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}

// Wrapper components to handle navigation
function RecordScreenWrapper({ onBack, onNavigateToAnalysis }: { onBack: () => void; onNavigateToAnalysis: () => void }) {
  const { colors, isDark } = useTheme();
  
  return (
    <GradientBackground>
      <View style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16, marginTop: 24, borderBottomWidth: 1, borderBottomColor: isDark ? '#333' : '#e0e0e0' }}>
          <TouchableOpacity onPress={onBack}>
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <View style={{ flex: 1, alignItems: 'center' }}>
            <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Record</Text>
          </View>
        </View>
        <RecordScreen navigation={{ goBack: onBack, navigate: (screen: string) => {
          if (screen === 'AnalysisResultScreen') {
            onNavigateToAnalysis();
          }
        }}} />
      </View>
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={onBack}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}

function HistoryScreenWrapper({ onBack }: { onBack: () => void }) {
  const { colors, isDark } = useTheme();
  
  return (
    <GradientBackground>
      <View style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16, marginTop: 24, borderBottomWidth: 1, borderBottomColor: isDark ? '#333' : '#e0e0e0' }}>
          <TouchableOpacity onPress={onBack}>
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <View style={{ flex: 1, alignItems: 'center' }}>
            <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>History & Insights</Text>
          </View>
        </View>
        <HistoryAndInsightsScreen navigation={{ goBack: onBack }} />
      </View>
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={onBack}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}

function SettingsScreenWrapper({ onBack, onNavigateToEditProfile }: { onBack: () => void; onNavigateToEditProfile: () => void }) {
  const { colors, isDark } = useTheme();
  
  return (
    <GradientBackground>
      <View style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16, marginTop: 24, borderBottomWidth: 1, borderBottomColor: isDark ? '#333' : '#e0e0e0' }}>
          <TouchableOpacity onPress={onBack}>
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <View style={{ flex: 1, alignItems: 'center' }}>
            <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Settings</Text>
          </View>
        </View>
        <SettingsScreen navigation={{ goBack: onBack, navigate: (screen: string) => {
          if (screen === 'EditBabyProfileScreen') {
            onNavigateToEditProfile();
          }
        }}} />
      </View>
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={onBack}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}

function EditProfileScreenWrapper({ onBack }: { onBack: () => void }) {
  const { colors, isDark } = useTheme();
  
  return (
    <GradientBackground>
      <View style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16, marginTop: 24, borderBottomWidth: 1, borderBottomColor: isDark ? '#333' : '#e0e0e0' }}>
          <TouchableOpacity onPress={onBack}>
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <View style={{ flex: 1, alignItems: 'center' }}>
            <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Edit Profile</Text>
          </View>
        </View>
        <EditBabyProfileScreen navigation={{ goBack: onBack }} />
      </View>
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={onBack}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}

function AnalysisScreenWrapper({ onBack, onNavigateToRecord }: { onBack: () => void; onNavigateToRecord: () => void }) {
  const { colors, isDark } = useTheme();
  
  return (
    <GradientBackground>
      <View style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16, marginTop: 24, borderBottomWidth: 1, borderBottomColor: isDark ? '#333' : '#e0e0e0' }}>
          <TouchableOpacity onPress={onBack}>
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <View style={{ flex: 1, alignItems: 'center' }}>
            <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Analysis Result</Text>
          </View>
        </View>
        <AnalysisResultScreen navigation={{ goBack: onBack, navigateToRecord: onNavigateToRecord }} />
      </View>
      
      {/* Bottom Navigation */}
      <View style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: isDark ? '#1a1a1a' : 'white',
        borderTopWidth: 1,
        borderTopColor: isDark ? '#333' : '#e0e0e0',
        paddingBottom: 12,
        paddingTop: 8,
      }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
          <TouchableOpacity 
            style={{ 
              alignItems: 'center', 
              padding: 12,
              borderRadius: 20,
              backgroundColor: 'transparent',
              minWidth: 50,
              minHeight: 50
            }}
            onPress={onBack}
          >
            <Ionicons 
              name="home" 
              size={24} 
              color={isDark ? '#ECEDEE' : '#333'} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </GradientBackground>
  );
}
