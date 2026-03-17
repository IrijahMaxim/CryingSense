import { useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useColorScheme as useSystemColorScheme } from 'react-native';

export function useAppColorScheme() {
  const systemColorScheme = useSystemColorScheme();
  const [colorScheme, setColorScheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    loadColorScheme();
  }, []);

  const loadColorScheme = async () => {
    try {
      const savedScheme = await AsyncStorage.getItem('colorScheme');
      if (savedScheme) {
        setColorScheme(JSON.parse(savedScheme));
      }
    } catch (error) {
      console.error('Error loading color scheme:', error);
    }
  };

  const toggleColorScheme = async (scheme: 'light' | 'dark') => {
    setColorScheme(scheme);
    await AsyncStorage.setItem('colorScheme', JSON.stringify(scheme));
  };

  return {
    colorScheme,
    setColorScheme: toggleColorScheme,
    isDarkMode: colorScheme === 'dark',
    systemColorScheme,
  };
}
