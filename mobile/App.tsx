import React from 'react';
import { View, StyleSheet } from 'react-native';
import { BabyProvider } from './src/context/BabyContext';
import { ThemeProvider } from './src/context/ThemeContext';
import HomeScreen from './src/screens/HomeScreen';

export default function App() {
  return (
    <BabyProvider>
      <ThemeProvider>
        <View style={styles.container}>
          <HomeScreen />
        </View>
      </ThemeProvider>
    </BabyProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
