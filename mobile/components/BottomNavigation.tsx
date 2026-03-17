import React from 'react';
import { View, TouchableOpacity, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useTheme } from '../src/context/ThemeContext';

export default function BottomNavigation() {
  const { colors, isDark } = useTheme();

  return (
    <View style={{
      position: 'absolute',
      bottom: 0,
      left: 0,
      right: 0,
      backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
      borderTopWidth: 1,
      borderTopColor: isDark ? '#333' : '#e0e0e0',
      paddingHorizontal: 20,
      paddingVertical: 12,
      flexDirection: 'row',
      justifyContent: 'space-around',
      alignItems: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: -2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 5,
    }}>
      
      {/* Home */}
      <TouchableOpacity
        style={{ alignItems: 'center', justifyContent: 'center' }}
        onPress={() => router.push('/')}
      >
        <Ionicons 
          name="home" 
          size={24} 
          color={colors.text} 
        />
      </TouchableOpacity>

      {/* Recording */}
      <TouchableOpacity
        style={{ alignItems: 'center', justifyContent: 'center' }}
        onPress={() => router.push('/record')}
      >
        <Ionicons 
          name="mic" 
          size={24} 
          color={colors.text} 
        />
      </TouchableOpacity>

      {/* History */}
      <TouchableOpacity
        style={{ alignItems: 'center', justifyContent: 'center' }}
        onPress={() => router.push('/history')}
      >
        <Ionicons 
          name="time" 
          size={24} 
          color={colors.text} 
        />
      </TouchableOpacity>

      {/* Settings */}
      <TouchableOpacity
        style={{ alignItems: 'center', justifyContent: 'center' }}
        onPress={() => router.push('/settings')}
      >
        <Ionicons 
          name="settings" 
          size={24} 
          color={colors.text} 
        />
      </TouchableOpacity>
    </View>
  );
}
