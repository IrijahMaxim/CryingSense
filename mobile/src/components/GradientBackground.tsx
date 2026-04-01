import React from 'react';
import { View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../context/ThemeContext';

interface GradientBackgroundProps {
  children: React.ReactNode;
  style?: any;
}

export default function GradientBackground({ children, style }: GradientBackgroundProps) {
  const { isDark } = useTheme();
  
  return (
    <LinearGradient
      colors={isDark ? ['#0a0a0f', '#1a1a3e'] : ['#FFE4E9', '#E6F3FF']} // Dark mode: very dark blue to dark blue, Light mode: very light pink to very light blue
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={[styles.gradient, style]}
    >
      {children}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: {
    flex: 1,
  },
});
