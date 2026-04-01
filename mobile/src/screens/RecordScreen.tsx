import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';
import styles from '../styles/RecordScreenStyles';
import GradientBackground from '../components/GradientBackground';

export default function RecordScreen({ navigation }: any) {
  const [isRecording, setIsRecording] = useState(false);
  const [cryDetected, setCryDetected] = useState(false);
  const { isDark } = useTheme();

  // Simulate cry detection
  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        // Simulate random cry detection
        const detected = Math.random() > 0.7;
        if (detected) {
          setCryDetected(true);
          Alert.alert(
            'Cry Detected!',
            'Baby cry detected. Analysis complete.',
            [{ text: 'OK', onPress: () => setCryDetected(false) }]
          );
        }
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isRecording]);

  const startRecording = () => {
    setIsRecording(true);
    setCryDetected(false);
  };

  const stopRecording = () => {
    setIsRecording(false);
    setCryDetected(false);
    Alert.alert('Recording Stopped', 'Analyzing the cry...');
    setTimeout(() => {
      if (navigation && navigation.navigate) {
        navigation.navigate('AnalysisResultScreen');
      }
    }, 1500);
  };

  const navigateToAnalysis = () => {
    if (navigation && navigation.navigate) {
      navigation.navigate('AnalysisResultScreen');
    }
  };

  return (
    <View style={{ flex: 1 }}>
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 20 }}>
          {/* Recording Status */}
          <View style={{
            backgroundColor: isDark ? '#2a2a2a' : 'white',
            padding: 30,
            borderRadius: 60,
            alignItems: 'center',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: isDark ? 0.3 : 0.1,
            shadowRadius: 8,
            elevation: 8,
            marginBottom: 15,
            width: 120,
            height: 120,
            alignSelf: 'center'
          }}>
            <Ionicons 
              name="mic" 
              size={60} 
              color={isRecording ? '#FF6347' : (isDark ? '#999' : '#666')} 
            />
          </View>

          {/* Cry Detection Alert */}
          {cryDetected && (
            <View style={{
              backgroundColor: '#4CAF50',
              padding: 16,
              borderRadius: 12,
              alignItems: 'center',
              marginTop: 20
            }}>
              <Ionicons name="checkmark-circle" size={32} color="white" style={{ marginBottom: 8 }} />
              <Text style={{ fontSize: 14, fontWeight: '600', color: 'white', textAlign: 'center' }}>
                Cry Detected!
              </Text>
            </View>
          )}

          {/* Control Buttons */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-around', width: '100%' }}>
            <TouchableOpacity
              style={{
                backgroundColor: isRecording ? '#FF6347' : '#4CAF50',
                padding: 16,
                borderRadius: 12,
                alignItems: 'center',
                marginTop: 30,
                flexDirection: 'row',
                justifyContent: 'center',
                minWidth: 150,
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.2,
                shadowRadius: 4,
                elevation: 4,
              }}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <Ionicons 
                name={isRecording ? "stop" : "play"} 
                size={24} 
                color="white" 
                style={{ marginRight: 8 }} 
              />
              <Text style={{ color: 'white', fontWeight: '600', fontSize: 14 }}>
                {isRecording ? 'Stop' : 'Start'}
              </Text>
            </TouchableOpacity>
          </View>

          {/* Test Analysis Button */}
          <TouchableOpacity
            style={{
              backgroundColor: '#60A5FA',
              padding: 16,
              borderRadius: 12,
              alignItems: 'center',
              marginTop: 30,
              flexDirection: 'row',
              justifyContent: 'center',
              minWidth: 200,
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.2,
              shadowRadius: 4,
              elevation: 4,
            }}
            onPress={navigateToAnalysis}>
            <Ionicons name="analytics" size={24} color="white" style={{ marginRight: 8 }} />
            <Text style={{ color: 'white', fontWeight: '600', fontSize: 14 }}>
              View Analysis Result
            </Text>
          </TouchableOpacity>
        </View>
    </View>
  );
}
