import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useTheme } from '../src/context/ThemeContext';
import { useBaby } from '../src/context/BabyContext';
import BottomNavigation from '../components/BottomNavigation';

// IoT Service Interface
interface IoTService {
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<AnalysisResult>;
  isRecording: boolean;
}

// Mock IoT Service (replace with real IoT SDK)
const mockIoTService: IoTService = {
  isRecording: false,
  
  async startRecording() {
    console.log('🎤 IoT: Starting recording...');
    this.isRecording = true;
    // Simulate IoT connection
    await new Promise(resolve => setTimeout(resolve, 1000));
  },
  
  async stopRecording(): Promise<AnalysisResult> {
    console.log('⏹️ IoT: Stopping recording...');
    this.isRecording = false;
    
    // Simulate IoT analysis processing
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock analysis result from IoT
    return {
      main: { condition: 'Hungry', confidence: 0.92 },
      suggestedActions: [
        'Try feeding your baby',
        'Check feeding schedule',
        'Feed in a calm environment',
      ],
      otherPossibilities: [
        { condition: 'Sleepy', confidence: 0.35 },
        { condition: 'Uncomfortable', confidence: 0.2 },
      ],
      timestamp: new Date(),
      duration: 15,
      confidence: 92
    };
  }
};

type AnalysisResult = {
  main: { condition: string; confidence: number };
  suggestedActions: string[];
  otherPossibilities: { condition: string; confidence: number }[];
  timestamp: Date;
  duration: number;
  confidence: number;
};

export default function RecordScreen() {
  const { colors, isDark } = useTheme();
  const { addCryEvent } = useBaby();
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      setIsProcessing(true);
      try {
        const analysisResult = await mockIoTService.stopRecording();
        
        // Add to history
        const newEvent = {
          id: Date.now().toString(),
          category: analysisResult.main.condition,
          timestamp: analysisResult.timestamp,
          need: analysisResult.main.condition,
          confidence: analysisResult.confidence,
          duration: analysisResult.duration
        };
        addCryEvent(newEvent);
        
        // Navigate to analysis results
        router.push({
          pathname: '/analysis-result',
          params: { 
            analysisData: JSON.stringify(analysisResult) 
          }
        });
        
        setIsRecording(false);
      } catch (error) {
        Alert.alert('Error', 'Failed to stop recording. Please try again.');
        console.error('IoT stop error:', error);
      } finally {
        setIsProcessing(false);
      }
    } else {
      // Start recording
      try {
        await mockIoTService.startRecording();
        setIsRecording(true);
        console.log('🎤 Recording started via IoT');
      } catch (error) {
        Alert.alert('Error', 'Failed to start recording. Please check IoT connection.');
        console.error('IoT start error:', error);
      }
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      {/* Header */}
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, marginBottom: 20 }}>
        <View style={{ width: 28 }} />
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Recording</Text>
        <View style={{ width: 28 }} />
      </View>

      {/* Recording Interface - IoT Connected */}
      <View style={{ 
        flex: 1, 
        justifyContent: 'center', 
        alignItems: 'center',
        paddingHorizontal: 16
      }}>
        <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text, marginBottom: 40, textAlign: 'center' }}>
          Baby Cry Detector
        </Text>
        
        <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 20, textAlign: 'center' }}>
          {isRecording ? '🔴 IoT Recording Active' : '🔵 IoT Ready'}
        </Text>
        
        <TouchableOpacity 
          onPress={toggleRecording}
          disabled={isProcessing}
          style={{ 
            width: 200, 
            height: 200, 
            borderRadius: 100, 
            backgroundColor: isProcessing ? '#999' : (isRecording ? '#ef4444' : '#60A5FA'),
            justifyContent: 'center',
            alignItems: 'center',
            marginBottom: 40,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.3,
            shadowRadius: 8,
            elevation: 8,
            opacity: isProcessing ? 0.6 : 1
          }}
        >
          <Ionicons 
            name={isProcessing ? "sync" : (isRecording ? "stop" : "mic")} 
            size={80} 
            color="white" 
            style={isProcessing ? { transform: [{ rotate: '360deg' }] } : {}}
          />
        </TouchableOpacity>

        <Text style={{ fontSize: 18, color: colors.text, textAlign: 'center', marginBottom: 20 }}>
          {isProcessing ? 'Processing...' : (isRecording ? 'Tap to stop recording' : 'Tap to start recording')}
        </Text>

        <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
          Connected to IoT Device • Real-time Analysis
        </Text>
      </View>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
