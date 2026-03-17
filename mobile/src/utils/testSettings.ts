// Test AsyncStorage functionality
import AsyncStorage from '@react-native-async-storage/async-storage';

// Test setting and getting preferences
const testSettingsPersistence = async () => {
  try {
    // Test setting values
    await AsyncStorage.setItem('notifications', JSON.stringify(true));
    await AsyncStorage.setItem('sound', JSON.stringify(false));
    await AsyncStorage.setItem('darkMode', JSON.stringify(true));
    
    // Test getting values
    const notifications = await AsyncStorage.getItem('notifications');
    const sound = await AsyncStorage.getItem('sound');
    const darkMode = await AsyncStorage.getItem('darkMode');
    
    console.log('Settings persistence test results:');
    console.log('Notifications:', notifications ? JSON.parse(notifications) : null);
    console.log('Sound:', sound ? JSON.parse(sound) : null);
    console.log('Dark Mode:', darkMode ? JSON.parse(darkMode) : null);
    
    return {
      notifications: notifications ? JSON.parse(notifications) : null,
      sound: sound ? JSON.parse(sound) : null,
      darkMode: darkMode ? JSON.parse(darkMode) : null,
    };
  } catch (error) {
    console.error('Settings persistence test failed:', error);
    return null;
  }
};

export default testSettingsPersistence;
