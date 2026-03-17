import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Image } from 'react-native';
import { useBaby } from '../context/BabyContext';
import styles from '../styles/AppStyles';
import * as ImagePicker from 'expo-image-picker';

export default function EditBabyProfileScreen({ navigation }: any) {
  const { profile, setProfile } = useBaby();
  const [name, setName] = useState(profile?.name || '');
  const [ageMonths, setAgeMonths] = useState(profile?.ageMonths?.toString() || '');
  const [photo, setPhoto] = useState(profile?.photo || null);

  const saveProfile = () => {
    setProfile({
      name: name.trim() || 'Baby',
      ageMonths: parseInt(ageMonths) || 0,
      photo,
    });
    navigation.goBack();
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled && result.assets.length > 0) {
      setPhoto(result.assets[0].uri);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.sectionTitle}>Edit Baby Profile</Text>

      <TouchableOpacity onPress={pickImage} style={styles.card}>
        {photo ? (
          <Image source={{ uri: photo }} style={{ width: 100, height: 100, borderRadius: 50 }} />
        ) : (
          <Image
            source={require('../assets/baby_placeholder.png')}
            style={{ width: 100, height: 100, borderRadius: 50 }}
          />
        )}
      </TouchableOpacity>

      <Text style={styles.insightTitle}>Name</Text>
      <TextInput
        style={styles.textInput}
        value={name}
        onChangeText={setName}
        placeholder="Enter baby name"
      />

      <Text style={styles.insightTitle}>Age (months)</Text>
      <TextInput
        style={styles.textInput}
        value={ageMonths}
        onChangeText={setAgeMonths}
        placeholder="Enter age in months"
        keyboardType="numeric"
      />

      <TouchableOpacity style={styles.saveButton} onPress={saveProfile}>
        <Text style={styles.saveButtonText}>Save</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}