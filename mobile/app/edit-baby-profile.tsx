import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, TextInput, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useBaby } from '../src/context/BabyContext';
import { useTheme } from '../src/context/ThemeContext';
import BottomNavigation from '../components/BottomNavigation';
import * as ImagePicker from 'expo-image-picker';

export default function EditBabyProfileScreen() {
  const { profile, setProfile } = useBaby();
  const { colors, isDark } = useTheme();
  const [name, setName] = useState(profile?.name || '');
  const [ageMonths, setAgeMonths] = useState(profile?.ageMonths?.toString() || '');
  const [photo, setPhoto] = useState(profile?.photo || null);

  const saveProfile = () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter a baby name');
      return;
    }

    if (!ageMonths.trim() || isNaN(Number(ageMonths)) || Number(ageMonths) < 0) {
      Alert.alert('Error', 'Please enter a valid age in months');
      return;
    }

    setProfile({
      ...profile,
      name: name.trim(),
      ageMonths: Number(ageMonths),
      photo: photo
    });

    Alert.alert('Success', 'Baby profile updated successfully!');
    router.back();
  };

  const pickImage = async () => {
    // Request permission
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission denied', 'Sorry, we need camera roll permissions to make this work!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.5,
    });

    if (!result.canceled && result.assets && result.assets[0]) {
      setPhoto(result.assets[0].uri);
    }
  };

  return (
    <ScrollView style={{ flex: 1, backgroundColor: colors.background, paddingHorizontal: 16, paddingTop: 12 }}>
      {/* Header */}
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="close" size={28} color={colors.text} />
        </TouchableOpacity>
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Edit Baby Profile</Text>
        <View style={{ width: 28 }} />
      </View>

      {/* First Card - Baby Picture with Avatar Selection */}
      <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
        <View style={{ alignItems: 'center' }}>
          {/* Baby Picture in Center */}
          <TouchableOpacity onPress={pickImage} style={{ marginBottom: 20 }}>
            {photo ? (
              <Image 
                source={{ uri: photo }} 
                style={{ width: 120, height: 120, borderRadius: 60 }} 
              />
            ) : (
              <Image
                source={require('../assets/baby_placeholder.png')}
                style={{ width: 120, height: 120, borderRadius: 60, marginBottom: 12 }}
              />
            )}
          </TouchableOpacity>

          <TouchableOpacity style={{ backgroundColor: '#60A5FA', padding: 12, borderRadius: 8, alignItems: 'center' }} onPress={pickImage}>
            <Text style={{ color: 'white', fontSize: 16, fontWeight: '600' }}>Choose Photo</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Second Card - Baby Name */}
      <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
        <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 8 }}>Baby Name</Text>
        <TextInput
          style={{ backgroundColor: isDark ? '#333' : '#f5f5f5', color: colors.text, padding: 12, borderRadius: 8, borderWidth: 1, borderColor: isDark ? '#555' : '#ddd', fontSize: 14 }}
          value={name}
          onChangeText={setName}
          placeholder="Enter baby name"
          placeholderTextColor={isDark ? '#999' : '#666'}
        />
      </View>

      {/* Third Card - Baby Age */}
      <View style={{ backgroundColor: isDark ? '#2a2a2a' : 'white', padding: 16, borderRadius: 8, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: isDark ? 0.3 : 0.1, shadowRadius: 2, elevation: 2 }}>
        <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 8 }}>Age (months)</Text>
        <TextInput
          style={{ backgroundColor: isDark ? '#333' : '#f5f5f5', color: colors.text, padding: 12, borderRadius: 8, borderWidth: 1, borderColor: isDark ? '#555' : '#ddd', fontSize: 14 }}
          value={ageMonths}
          onChangeText={setAgeMonths}
          placeholder="Enter age in months"
          placeholderTextColor={isDark ? '#999' : '#666'}
          keyboardType="numeric"
        />
      </View>

      <TouchableOpacity style={{ backgroundColor: '#60A5FA', padding: 14, borderRadius: 8, alignItems: 'center', marginVertical: 20 }} onPress={saveProfile}>
        <Text style={{ color: 'white', fontSize: 16, fontWeight: '600' }}>Save</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}
