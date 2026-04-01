import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Switch, Linking, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useBaby } from '../context/BabyContext';
import { useTheme } from '../context/ThemeContext';
import styles from '../styles/AppStyles';
import GradientBackground from '../components/GradientBackground';

export default function SettingsScreen({ navigation }: any) {
  const { profile } = useBaby();
  const { theme, setTheme, isDark } = useTheme();

  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(false); // Default to false

  // TODO: Load settings when AsyncStorage is properly installed
  // TODO: Save settings when AsyncStorage is properly installed

  // Mock functions for now
  const toggleNotifications = async (value: boolean) => {
    setNotificationsEnabled(value);
    Alert.alert('Settings', `Notifications ${value ? 'enabled' : 'disabled'}`);
  };

  const toggleSound = async (value: boolean) => {
    setSoundEnabled(value);
    Alert.alert('Settings', `Sound ${value ? 'enabled' : 'disabled'}`);
  };

  const toggleDarkMode = async (value: boolean) => {
    const newTheme = value ? 'dark' : 'light';
    setTheme(newTheme);
    Alert.alert('Settings', `Theme changed to ${newTheme} mode`);
  };

  const handleHelpSupport = () => {
    Alert.alert(
      'Help & Support',
      'Settings functionality will be available once dependencies are resolved.',
      [{ text: 'OK', style: 'default' }]
    );
  };

  const handlePrivacyPolicy = () => {
    Alert.alert(
      'Privacy Policy',
      'Privacy policy will be available once dependencies are resolved.',
      [{ text: 'OK', style: 'default' }]
    );
  };

  return (
    <ScrollView style={{ backgroundColor: 'transparent', flex: 1 }}>
      {/* Baby Profile */}
      <Text style={{ fontSize: 18, fontWeight: 'bold', color: isDark ? '#ECEDEE' : '#333', marginBottom: 12, marginTop: 8 }}>Baby Profile</Text>
      <TouchableOpacity
        style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 12, 
          marginBottom: 15, 
          width: '95%', 
          alignSelf: 'center' 
        }}
        onPress={() => {
          if (navigation && navigation.navigate) {
            navigation.navigate('EditBabyProfileScreen');
          }
        }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          {profile?.photo ? (
            <Image
              source={{ uri: profile.photo }}
              style={{ width: 50, height: 50, borderRadius: 12, marginRight: 12 }}
            />
          ) : (
            <View style={{ width: 50, height: 50, borderRadius: 25, backgroundColor: isDark ? '#333' : '#e0e0e0', marginRight: 12, justifyContent: 'center', alignItems: 'center' }}>
              <Ionicons name="person" size={24} color={isDark ? '#999' : '#666'} />
            </View>
          )}
          <View style={{ flex: 1 }}>
            <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>{profile.name}</Text>
            <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>{profile.ageMonths} months old</Text>
          </View>
        </View>
        <Ionicons name="chevron-forward" size={24} color={isDark ? '#999' : '#666'} />
      </TouchableOpacity>

      {/* Preferences */}
      <Text style={{ fontSize: 18, fontWeight: 'bold', color: isDark ? '#ECEDEE' : '#333', marginBottom: 12, marginTop: 8 }}>Preferences</Text>
      <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 12, marginBottom: 6, width: '95%', alignSelf: 'center' }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
          <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>Notifications</Text>
          <Switch value={notificationsEnabled} onValueChange={toggleNotifications} />
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
          <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>Sound Alerts</Text>
          <Switch value={soundEnabled} onValueChange={toggleSound} />
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15 }}>
          <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>Dark Mode</Text>
          <Switch value={isDark} onValueChange={toggleDarkMode} />
        </View>
      </View>

      {/* Support */}
      <Text style={{ fontSize: 18, fontWeight: 'bold', color: isDark ? '#ECEDEE' : '#333', marginBottom: 12, marginTop: 8 }}>Support & Information</Text>
      <TouchableOpacity
        style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 12, 
          marginBottom: 15, 
          width: '95%', 
          alignSelf: 'center' 
        }}
        onPress={() => handleHelpSupport()}>
        <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>Help & Support</Text>
        <Ionicons name="chevron-forward" size={24} color={isDark ? '#999' : '#666'} />
      </TouchableOpacity>
      <TouchableOpacity
        style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 12, 
          marginBottom: 80, 
          width: '95%', 
          alignSelf: 'center' 
        }}
        onPress={() => handlePrivacyPolicy()}>
        <Text style={{ fontWeight: 'normal', color: isDark ? '#ECEDEE' : '#333' }}>Privacy Policy</Text>
        <Ionicons name="chevron-forward" size={24} color={isDark ? '#999' : '#666'} />
      </TouchableOpacity>
    </ScrollView>
  );
}
