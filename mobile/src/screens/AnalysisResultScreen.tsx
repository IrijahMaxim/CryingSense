import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, Dimensions } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';
import BottomNavigation from '../../components/BottomNavigation';

// Analysis Result Type from IoT
type AnalysisResult = {
  main: { condition: string; confidence: number };
  suggestedActions: string[];
  otherPossibilities: { condition: string; confidence: number }[];
  timestamp: Date;
  duration: number;
  confidence: number;
};

// Screen width for progress bar
const screenWidth = Dimensions.get('window').width - 32; // minus container padding

// Progress bar component
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

export default function AnalysisResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const { colors, isDark } = useTheme();
  
  // Get analysis data from IoT or use mock
  const result: AnalysisResult = params.analysisData 
    ? JSON.parse(params.analysisData as string)
    : {
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
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <ScrollView style={{ paddingHorizontal: 16, paddingTop: 12, paddingBottom: 80 }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text }}>Analysis Result</Text>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="close" size={28} color={colors.text} />
          </TouchableOpacity>
        </View>

        {/* IoT Status */}
        <View style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 16, 
          borderRadius: 12, 
          marginBottom: 20, 
          alignItems: 'center',
          shadowColor: '#000', 
          shadowOffset: { width: 0, height: 2 }, 
          shadowOpacity: isDark ? 0.3 : 0.1, 
          shadowRadius: 4, 
          elevation: 3 
        }}>
          <Text style={{ fontSize: 14, color: '#4CAF50', fontWeight: '600', marginBottom: 4 }}>
            🎯 IoT Analysis Complete
          </Text>
          <Text style={{ fontSize: 12, color: isDark ? '#999' : '#666' }}>
            Real-time cry detection and analysis
          </Text>
        </View>

        {/* Main Result Card */}
        <View style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 20, 
          borderRadius: 12, 
          marginBottom: 20, 
          alignItems: 'center',
          shadowColor: '#000', 
          shadowOffset: { width: 0, height: 2 }, 
          shadowOpacity: isDark ? 0.3 : 0.1, 
          shadowRadius: 4, 
          elevation: 3 
        }}>
          <Ionicons 
            name={result.main.condition === 'Hungry' ? 'restaurant' : 
                 result.main.condition === 'Sleepy' ? 'bed' : 
                 'alert-circle'} 
            size={48} 
            color="#FF6347" 
            style={{ marginBottom: 12 }} 
          />
          <Text style={{ fontSize: 20, fontWeight: '600', color: colors.text, marginBottom: 8 }}>
            Baby is {result.main.condition}
          </Text>
          <Text style={{ fontSize: 16, color: isDark ? '#999' : '#666', marginBottom: 12 }}>
            Confidence: {(result.main.confidence * 100).toFixed(0)}%
          </Text>
          <ProgressBar progress={result.main.confidence} />
        </View>

        {/* Recording Details */}
        <View style={{ 
          backgroundColor: isDark ? '#2a2a2a' : 'white', 
          padding: 16, 
          borderRadius: 12, 
          marginBottom: 20,
          shadowColor: '#000', 
          shadowOffset: { width: 0, height: 2 }, 
          shadowOpacity: isDark ? 0.3 : 0.1, 
          shadowRadius: 4, 
          elevation: 3 
        }}>
          <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 12 }}>
            Recording Details
          </Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <View>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 4 }}>Duration</Text>
              <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>{result.duration}s</Text>
            </View>
            <View>
              <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 4 }}>Time</Text>
              <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>
                {new Date(result.timestamp).toLocaleTimeString()}
              </Text>
            </View>
          </View>
        </View>

        {/* Suggested Actions */}
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: colors.text, marginBottom: 16 }}>
          Suggested Actions
        </Text>
        {result.suggestedActions.map((action, index) => (
          <View 
            key={index} 
            style={{ 
              backgroundColor: isDark ? '#2a2a2a' : 'white', 
              padding: 16, 
              borderRadius: 12, 
              marginBottom: 12,
              shadowColor: '#000', 
              shadowOffset: { width: 0, height: 2 }, 
              shadowOpacity: isDark ? 0.3 : 0.1, 
              shadowRadius: 4, 
              elevation: 3 
            }}
          >
            <Text style={{ fontSize: 16, color: colors.text }}>{action}</Text>
          </View>
        ))}

        {/* Other Possibilities */}
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: colors.text, marginBottom: 16 }}>
          Other Possibilities
        </Text>
        {result.otherPossibilities.map((option, index) => (
          <View 
            key={index} 
            style={{ 
              backgroundColor: isDark ? '#2a2a2a' : 'white', 
              padding: 16, 
              borderRadius: 12, 
              marginBottom: 12,
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
            <Text style={{ fontSize: 16, color: colors.text }}>{option.condition}</Text>
            <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666' }}>
              {(option.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        ))}

        {/* Return to Home Button */}
        <TouchableOpacity 
          style={{ 
            backgroundColor: '#60A5FA', 
            padding: 16, 
            borderRadius: 12, 
            alignItems: 'center', 
            marginTop: 20,
            marginBottom: 20
          }} 
          onPress={() => router.push('/')}
        >
          <Text style={{ fontSize: 16, fontWeight: '600', color: 'white' }}>Return to Home</Text>
        </TouchableOpacity>
      </ScrollView>
      
      {/* Bottom Navigation */}
      <BottomNavigation />
    </View>
  );
}
