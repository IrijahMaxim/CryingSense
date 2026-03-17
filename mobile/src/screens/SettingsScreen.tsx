import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Switch, Linking, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { HomeStackParamList } from '../navigation/HomeStack';
import { RootStackParamList } from '../../App';
import { CompositeNavigationProp } from '@react-navigation/native';
import { useBaby } from '../context/BabyContext';
import styles from '../styles/AppStyles';

type SettingsScreenNavigationProp = CompositeNavigationProp<
  NativeStackNavigationProp<HomeStackParamList, 'SettingsScreen'>,
  NativeStackNavigationProp<RootStackParamList>
>;

export default function SettingsScreen() {
  const navigation = useNavigation<SettingsScreenNavigationProp>();
  const { profile } = useBaby();

  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  // Load saved settings
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const notif = await AsyncStorage.getItem('notifications');
    const sound = await AsyncStorage.getItem('sound');
    const dark = await AsyncStorage.getItem('darkMode');

    if (notif !== null) setNotificationsEnabled(JSON.parse(notif));
    if (sound !== null) setSoundEnabled(JSON.parse(sound));
    if (dark !== null) setDarkMode(JSON.parse(dark));
  };

  // Save settings
  const toggleNotifications = async (value: boolean) => {
    setNotificationsEnabled(value);
    await AsyncStorage.setItem('notifications', JSON.stringify(value));
  };

  const toggleSound = async (value: boolean) => {
    setSoundEnabled(value);
    await AsyncStorage.setItem('sound', JSON.stringify(value));
  };

  const toggleDarkMode = async (value: boolean) => {
    setDarkMode(value);
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
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={28} color="#333" />
        </TouchableOpacity>
        <Text style={styles.sectionTitle}>Settings</Text>
        <View style={{ width: 28 }} />
      </View>

      {/* Baby Profile */}
      <Text style={styles.sectionTitle}>Baby Profile</Text>
      <TouchableOpacity
        style={styles.card}
        onPress={() => navigation.navigate('EditBabyProfileScreen')}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          {profile?.photo ? (
            <Image
              source={{ uri: profile.photo }}
              style={{ width: 50, height: 50, borderRadius: 25, marginRight: 12 }}
            />
          ) : (
            <Image
              source={require('../../assets/baby_placeholder.png')}
              style={{ width: 50, height: 50, borderRadius: 25, marginRight: 12 }}
            />
          )}
          <View>
            <Text style={styles.cardTitle}>{profile?.name}</Text>
            <Text>{profile?.ageMonths} months old</Text>
          </View>
        </View>
        <Ionicons name="chevron-forward" size={24} color="#666" />
      </TouchableOpacity>

      {/* Preferences */}
      <Text style={styles.sectionTitle}>Preferences</Text>

      <View style={styles.preferenceRow}>
        <Text style={styles.preferenceText}>Notifications</Text>
        <Switch value={notificationsEnabled} onValueChange={toggleNotifications} />
      </View>

      <View style={styles.preferenceRow}>
        <Text style={styles.preferenceText}>Sound Alerts</Text>
        <Switch value={soundEnabled} onValueChange={toggleSound} />
      </View>

      <View style={styles.preferenceRow}>
        <Text style={styles.preferenceText}>Dark Mode</Text>
        <Switch value={darkMode} onValueChange={toggleDarkMode} />
      </View>

      {/* Support */}
      <Text style={styles.sectionTitle}>Support & Information</Text>

      <TouchableOpacity style={styles.card} onPress={() => handleHelpSupport()}>
        <Text style={styles.cardTitle}>Help & Support</Text>
        <Ionicons name="chevron-forward" size={24} color="#666" />
      </TouchableOpacity>

      <TouchableOpacity style={styles.card} onPress={() => handlePrivacyPolicy()}>
        <Text style={styles.cardTitle}>Privacy Policy</Text>
        <Ionicons name="chevron-forward" size={24} color="#666" />
      </TouchableOpacity>
    </ScrollView>
  );
}
