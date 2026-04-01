import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Colors } from '../constants/theme';

type Theme = 'light' | 'dark';

type ThemeContextType = {
  theme: Theme;
  colors: typeof Colors.light | typeof Colors.dark;
  setTheme: (theme: Theme) => void;
  isDark: boolean;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('light');

  useEffect(() => {
    loadTheme();
  }, []);

  const loadTheme = async () => {
    try {
      const savedTheme = await AsyncStorage.getItem('theme');
      if (savedTheme) {
        setThemeState(JSON.parse(savedTheme));
      }
    } catch (error) {
      console.error('Error loading theme:', error);
    }
  };

  const setTheme = async (newTheme: Theme) => {
    setThemeState(newTheme);
    await AsyncStorage.setItem('theme', JSON.stringify(newTheme));
  };

  const colors = theme === 'dark' ? Colors.dark : Colors.light;
  const isDark = theme === 'dark';

  return (
    <ThemeContext.Provider value={{ theme, colors, setTheme, isDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
