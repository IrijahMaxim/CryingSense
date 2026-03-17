import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Switch, Linking, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Notifications from 'expo-notifications';
import { useBaby } from '../src/context/BabyContext';
import { useTheme } from '../src/context/ThemeContext';
import BottomNavigation from '../components/BottomNavigation';

export default function SettingsScreen() {
  const { profile } = useBaby();
  const { theme, colors, setTheme, isDark } = useTheme();

  const [notificationsEnabled, setNotificationsEnabled] = React.useState(true);
  const [soundEnabled, setSoundEnabled] = React.useState(true);
  const [darkMode, setDarkMode] = React.useState(isDark);

  // Load saved settings
  React.useEffect(() => {
    loadSettings();
  }, []);

  React.useEffect(() => {
    setDarkMode(isDark);
  }, [isDark]);

  const loadSettings = async () => {
    try {
      const notif = await AsyncStorage.getItem('notifications');
      const sound = await AsyncStorage.getItem('sound');
      const dark = await AsyncStorage.getItem('darkMode');

      if (notif !== null) setNotificationsEnabled(JSON.parse(notif));
      if (sound !== null) setSoundEnabled(JSON.parse(sound));
      if (dark !== null) {
        const darkValue = JSON.parse(dark);
        setDarkMode(darkValue);
        if (darkValue !== isDark) {
          setTheme(darkValue ? 'dark' : 'light');
        }
      }
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  // Save settings
  const toggleNotifications = async (value: boolean) => {
    setNotificationsEnabled(value);
    await AsyncStorage.setItem('notifications', JSON.stringify(value));
    
    // Handle notification permissions
    if (value) {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please enable notifications in your device settings to receive alerts.');
        setNotificationsEnabled(false);
        await AsyncStorage.setItem('notifications', JSON.stringify(false));
      }
    } else {
      await Notifications.cancelAllScheduledNotificationsAsync();
    }
  };

  const toggleSound = async (value: boolean) => {
    setSoundEnabled(value);
    await AsyncStorage.setItem('sound', JSON.stringify(value));
    await AsyncStorage.setItem('soundEnabled', JSON.stringify(value));
  };

  const toggleDarkMode = async (value: boolean) => {
    setDarkMode(value);
    setTheme(value ? 'dark' : 'light');
    await AsyncStorage.setItem('darkMode', JSON.stringify(value));
  };

  const handleHelpSupport = () => {
    Alert.alert(
      'Help & Support',
      'How can we help you today?',
      [
        {
          text: 'Email Support',
          onPress: () => Linking.openURL('mailto:support@cryingsense.com'),
        },
        {
          text: 'User Guide',
          onPress: () => Linking.openURL('https://cryingsense.com/guide'),
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ]
    );
  };

  const handlePrivacyPolicy = () => {
    Alert.alert(
      'Privacy Policy',
      'View our Privacy Policy to understand how we protect your data.',
      [
        {
          text: 'Open Privacy Policy',
          onPress: () => Linking.openURL('https://cryingsense.com/privacy'),
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ]
    );
  };

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <ScrollView style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 0, paddingTop: 0, marginBottom: 20 }}>
          <View style={{ width: 28 }} />
          <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 12, marginTop: 8 }}>Settings</Text>
          <View style={{ width: 28 }} />
        </View>

        {/* Baby Profile */}
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 12, marginTop: 8 }}>Baby Profile</Text>
        <TouchableOpacity
          style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}
          onPress={() => router.push('/edit-baby-profile')}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              {profile?.photo ? (
                <Image
                  source={{ uri: profile.photo }}
                  style={{ width: 50, height: 50, borderRadius: 25, marginRight: 12 }}
                />
              ) : (
                <Image
                  source={require('../assets/baby_placeholder.png')}
                  style={{ width: 50, height: 50, borderRadius: 25, marginRight: 12 }}
                />
              )}
              <View>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>{profile?.name}</Text>
                <Text style={{ color: colors.text }}>{profile?.ageMonths} months old</Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={24} color="#666" />
          </View>
        </TouchableOpacity>

        {/* Preferences */}
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 12, marginTop: 8 }}>Preferences</Text>

        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, backgroundColor: isDark ? '#2a2a2a' : 'white', borderRadius: 8, marginBottom: 8, paddingHorizontal: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
          <Text style={{ fontSize: 16, color: colors.text }}>Notifications</Text>
          <Switch value={notificationsEnabled} onValueChange={toggleNotifications} />
        </View>

        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, backgroundColor: isDark ? '#2a2a2a' : 'white', borderRadius: 8, marginBottom: 8, paddingHorizontal: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
          <Text style={{ fontSize: 16, color: colors.text }}>Sound Alerts</Text>
          <Switch value={soundEnabled} onValueChange={toggleSound} />
        </View>

        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, backgroundColor: isDark ? '#2a2a2a' : 'white', borderRadius: 8, marginBottom: 8, paddingHorizontal: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
          <Text style={{ fontSize: 16, color: colors.text }}>Dark Mode</Text>
          <Switch value={darkMode} onValueChange={toggleDarkMode} />
        </View>

        {/* Support */}
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text, marginBottom: 12, marginTop: 8 }}>Support & Information</Text>

        <TouchableOpacity style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }} onPress={() => handleHelpSupport()}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>Help & Support</Text>
            <Ionicons name="chevron-forward" size={24} color="#666" />
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }} onPress={() => handlePrivacyPolicy()}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>Privacy Policy</Text>
            <Ionicons name="chevron-forward" size={24} color="#666" />
          </View>
        </TouchableOpacity>
      </ScrollView>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
