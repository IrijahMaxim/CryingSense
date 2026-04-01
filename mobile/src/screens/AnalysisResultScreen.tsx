import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Dimensions, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';
import GradientBackground from '../components/GradientBackground';

type FilterOption = 'All' | 'Today' | 'Yesterday' | 'Week';

// Analysis Result Type
type AnalysisResult = {
  main: { condition: string; confidence: number };
  suggestedActions: string[];
  otherPossibilities: { condition: string; confidence: number }[];
  timestamp: Date;
  duration: number;
  confidence: number;
};

type CryData = {
  timestamp: Date;
  confidence: number;
  condition: string;
  duration: number;
  deviceId?: string;
};

// Screen width for progress bar
const screenWidth = Dimensions.get('window').width - 32; // minus container padding

// IoT Actions - Removed for now to avoid errors
  // const scanForDevices = async () => {
  //   setIsScanning(true);
  //   try {
  //     const devices = await IoTService.scanForDevices();
  //     Alert.alert(
  //       'Device Scan Complete',
  //       `Found ${devices.length} CryingSense devices`,
  //       [{ text: 'OK', style: 'default' }]
  //     );
  //   } catch (error) {
  //     Alert.alert(
  //       'Scan Failed',
  //       'Could not scan for devices. Please check your network connection.',
  //       [{ text: 'OK', style: 'default' }]
  //     );
  //   } finally {
  //     setIsScanning(false);
  //   }
  // };
  
  // const sendTestCryData = async () => {
  //   const testData: CryData = {
  //     timestamp: new Date(),
  //     confidence: 0.95,
  //     condition: 'Hungry',
  //     duration: 12,
  //     deviceId: iotDevices.length > 0 ? iotDevices[0].id : undefined
  //   };
  //   
  //   const success = await IoTService.sendCryData(testData);
  //   if (success) {
  //     Alert.alert(
  //       'Data Sent',
  //       'Cry data successfully sent to IoT server',
  //       [{ text: 'OK', style: 'default' }]
  //     );
  //   } else {
  //     Alert.alert(
  //       'Send Failed',
  //       'Failed to send cry data to server. Please check connection.',
  //       [{ text: 'OK', style: 'default' }]
  //     );
  //   }
  // };
const ProgressBar = ({ progress }: { progress: number }) => (
  <View
    style={{
      height: 10,
      width: screenWidth,
      backgroundColor: '#eee',
      borderRadius: 5,
      marginTop: 8,
    }}>
    <View
      style={{
        height: '100%',
        width: screenWidth * progress, // numeric width ✅
        backgroundColor: '#FF6347',
        borderRadius: 5,
      }}
    />
  </View>
);

export default function AnalysisResultScreen({ navigation }: any) {
  const { colors, isDark } = useTheme();
  const [filter, setFilter] = useState<FilterOption>('All');
  
  // Get analysis data from IoT or use mock
  const result: AnalysisResult = {
    main: { condition: 'Hungry', confidence: 0.92 },
    suggestedActions: [
      'Try feeding your baby',
      'Check feeding schedule',
      'Feed in a calm environment',
    ],
    otherPossibilities: [
      { condition: 'Sleepy', confidence: 0.35 },
      { condition: 'Discomfort', confidence: 0.2 },
    ],
    timestamp: new Date(),
    duration: 15,
    confidence: 92
  };

  return (
      <View style={{ flex: 1 }}>
        <ScrollView 
          style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}
          contentContainerStyle={{ paddingBottom: 100 }}
          showsVerticalScrollIndicator={true}
        >
          {/* Header */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          </View>

          {/* -------- Filter Options -------- */}
          <View style={{ 
            flexDirection: 'row', 
            marginBottom: 12,
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
            {(['All', 'Today', 'Yesterday', 'Week'] as FilterOption[]).map((option) => (
              <TouchableOpacity
                key={option}
                style={{
                  paddingVertical: 6,
                  paddingHorizontal: 12,
                  borderRadius: 12,
                  marginRight: 8,
                  backgroundColor: filter === option ? '#60A5FA' : (isDark ? '#2a2a2a' : '#f0f0f0'),
                }}
                onPress={() => setFilter(option)}
              >
                <Text style={{
                  fontSize: 11,
                  fontWeight: 'normal',
                  color: filter === option ? 'white' : (isDark ? '#ECEDEE' : '#333'),
                }}>
                  {option}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* IoT Status */}
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 12, 
            borderRadius: 12, 
            marginBottom: 12,
            width: '95%', alignSelf: 'center', 
            alignItems: 'center',
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 8 }}>IoT Device Status</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="wifi" size={20} color="#4CAF50" />
              <Text style={{ fontSize: 14, color: '#4CAF50', marginLeft: 8 }}>Connected</Text>
            </View>
          </View>

          {/* Main Result Card */}
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 20, 
            borderRadius: 12, 
            marginBottom: 15, 
            alignItems: 'center',
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Ionicons name="person" size={40} color="#60A5FA" style={{ marginBottom: 16 }} />
            <Text style={{ fontSize: 16, fontWeight: 'normal', color: colors.text, marginBottom: 8 }}>
              {result.main.condition}
            </Text>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', textAlign: 'center' }}>
              Confidence: {(result.main.confidence * 100).toFixed(0)}%
            </Text>
          </View>

          {/* Recording Details */}
          <View style={{ 
            backgroundColor: isDark ? '#2a2a2a' : 'white', 
            padding: 12, 
            borderRadius: 12, 
            marginBottom: 12,
            width: '95%', alignSelf: 'center',
            shadowColor: '#000', 
            shadowOffset: { width: 0, height: 2 }, 
            shadowOpacity: isDark ? 0.3 : 0.1, 
            shadowRadius: 4, 
            elevation: 3 
          }}>
            <Text style={{ fontSize: 14, fontWeight: 'normal', color: colors.text, marginBottom: 12 }}>Recording Details</Text>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 }}>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>Duration:</Text>
              <Text style={{ fontSize: 14, color: colors.text }}>{result.duration}s</Text>
            </View>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 }}>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>Time:</Text>
              <Text style={{ fontSize: 14, color: colors.text }}>{new Date().toLocaleTimeString()}</Text>
            </View>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>Date:</Text>
              <Text style={{ fontSize: 14, color: colors.text }}>{new Date().toLocaleDateString()}</Text>
            </View>
          </View>

          {/* Suggested Actions */}
          <View style={{ marginBottom: 20 }}>
            <Text style={{ fontSize: 14, fontWeight: 'normal', color: colors.text, marginBottom: 12 }}>Suggested Actions</Text>
            {result.suggestedActions.map((action: string, index: number) => (
              <View 
                key={index} 
                style={{ 
                  backgroundColor: isDark ? '#2a2a2a' : 'white', 
                  padding: 12, 
                  borderRadius: 12, 
                  marginBottom: 15,
                  width: '95%',
                  alignSelf: 'center',
                  shadowColor: '#000', 
                  shadowOffset: { width: 0, height: 2 }, 
                  shadowOpacity: isDark ? 0.3 : 0.1, 
                  shadowRadius: 4, 
                  elevation: 3 
                }}
              >
                <Text style={{ fontSize: 14, color: colors.text }}>{action}</Text>
              </View>
            ))}
          </View>

          {/* Other Possibilities */}
          <View style={{ marginBottom: 20 }}>
            <Text style={{ fontSize: 14, fontWeight: 'normal', color: colors.text, marginBottom: 12 }}>Other Possibilities</Text>
            {result.otherPossibilities.map((item: any, index: number) => (
              <View 
                key={index} 
                style={{ 
                  backgroundColor: isDark ? '#2a2a2a' : 'white', 
                  padding: 12, 
                  borderRadius: 12, 
                  marginBottom: 15,
                  width: '95%',
                  alignSelf: 'center',
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  shadowColor: '#000', 
                  shadowOffset: { width: 0, height: 2 }, 
                  shadowOpacity: isDark ? 0.3 : 0.1, 
                  shadowRadius: 4, 
                  elevation: 3 
                }}
              >
                <Text style={{ fontSize: 14, color: colors.text }}>{item.category}</Text>
                <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>{(item.confidence * 100).toFixed(0)}%</Text>
              </View>
            ))}
          </View>

          {/* Action Buttons */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: 20 }}>
            <TouchableOpacity
              style={{
                backgroundColor: '#60A5FA',
                padding: 16,
                borderRadius: 12,
                alignItems: 'center',
                flex: 1,
                marginRight: 8,
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.2,
                shadowRadius: 4,
                elevation: 4,
              }}
              onPress={() => {
                if (navigation && navigation.navigateToRecord) {
                  navigation.navigateToRecord();
                }
              }}
            >
              <Text style={{ color: 'white', fontWeight: '600', fontSize: 14 }}>Record Again</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={{
                backgroundColor: '#4CAF50',
                padding: 16,
                borderRadius: 12,
                alignItems: 'center',
                flex: 1,
                marginLeft: 8,
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.2,
                shadowRadius: 4,
                elevation: 4,
              }}
              onPress={() => {
                if (navigation && navigation.goBack) {
                  navigation.goBack();
                }
              }}
            >
              <Text style={{ color: 'white', fontWeight: '600', fontSize: 14 }}>Return to Home</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </View>
  );
}
