import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Image, Alert, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useBaby } from '../context/BabyContext';
import { useTheme } from '../context/ThemeContext';
import styles from '../styles/AppStyles';
import GradientBackground from '../components/GradientBackground';

export default function EditBabyProfileScreen({ navigation }: any) {
  const { profile, setProfile } = useBaby();
  const { colors, isDark } = useTheme();
  const [name, setName] = useState(profile?.name || '');
  const [ageMonths, setAgeMonths] = useState(profile?.ageMonths?.toString() || '');
  const [photo, setPhoto] = useState(profile?.photo || null);

  const saveProfile = () => {
    if (name.trim() && ageMonths.trim()) {
      setProfile({
        name: name.trim(),
        ageMonths: parseInt(ageMonths),
        photo: photo,
      });
      if (navigation && navigation.goBack) {
        navigation.goBack();
      }
    }
  };

  const pickImage = async () => {
    // TODO: Implement image picker when expo-image-picker is properly installed
    Alert.alert(
      'Image Picker',
      'Image picker will be available once dependencies are resolved.',
      [{ text: 'OK', style: 'default' }]
    );
    
    // Original image picker code (commented out):
    /*
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
    });

    if (!result.canceled) {
      setPhoto(result.assets[0].uri);
    }
    */
  };

  return (
      <ScrollView style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}>
        {/* Profile Form */}
        <View style={{
          backgroundColor: isDark ? '#2a2a2a' : 'white',
          padding: 16,
          borderRadius: 12,
          marginBottom: 15,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: isDark ? 0.3 : 0.1,
          shadowRadius: 4,
          elevation: 4,
          width: '95%',
          alignSelf: 'center',
        }}>
          {/* Photo */}
          <TouchableOpacity onPress={pickImage} style={{ alignItems: 'center', marginBottom: 20 }}>
            {photo ? (
              <Image
                source={{ uri: photo }}
                style={{ width: 100, height: 100, borderRadius: 50 }}
              />
            ) : (
              <Image
                source={require('../assets/baby_placeholder.png')}
                style={{ width: 100, height: 100, borderRadius: 50 }}
              />
            )}
            <View style={{ position: 'absolute', bottom: 0, right: 0, backgroundColor: '#60A5FA', borderRadius: 15, padding: 4 }}>
              <Ionicons name="camera" size={16} color="white" />
            </View>
          </TouchableOpacity>

          <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 20 }}>Baby Information</Text>

          <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 8 }}>Name</Text>
          <TextInput
            style={{
              backgroundColor: isDark ? 'rgba(26, 26, 26, 0.3)' : '#f5f5f5',
              color: colors.text,
              borderWidth: 1,
              borderColor: isDark ? '#333' : '#e0e0e0',
              borderRadius: 8,
              padding: 12,
              fontSize: 14,
              marginBottom: 15,
            }}
            placeholder="Enter baby's name"
            placeholderTextColor={isDark ? '#999' : '#666'}
            value={name}
            onChangeText={setName}
          />

          <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 8 }}>Age (months)</Text>
          <TextInput
            style={{
              backgroundColor: isDark ? 'rgba(26, 26, 26, 0.3)' : '#f5f5f5',
              color: colors.text,
              borderWidth: 1,
              borderColor: isDark ? '#333' : '#e0e0e0',
              borderRadius: 8,
              padding: 12,
              fontSize: 14,
              marginBottom: 15,
            }}
            placeholder="Enter baby's age in months"
            placeholderTextColor={isDark ? '#999' : '#666'}
            value={ageMonths}
            onChangeText={setAgeMonths}
            keyboardType="numeric"
          />
        </View>

        {/* Save Button */}
        <TouchableOpacity
          style={{
            backgroundColor: '#60A5FA',
            padding: 16,
            borderRadius: 12,
            alignItems: 'center',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.2,
            shadowRadius: 4,
            elevation: 4,
          }}
          onPress={saveProfile}
        >
          <Text style={{ color: 'white', fontWeight: '600', fontSize: 14 }}>Save Profile</Text>
        </TouchableOpacity>
      </ScrollView>
  );
}