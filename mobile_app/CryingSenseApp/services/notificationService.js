import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

// Configure notification behavior
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// Request notification permissions
export const requestNotificationPermissions = async () => {
  try {
    const { status } = await Notifications.requestPermissionsAsync();
    
    if (status !== 'granted') {
      console.log('Notification permissions not granted');
      return false;
    }
    
    return true;
  } catch (error) {
    console.error('Error requesting notification permissions:', error);
    return false;
  }
};

// Send local notification
export const sendLocalNotification = async (title, body, data = {}) => {
  try {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: true,
      },
      trigger: null, // Send immediately
    });
  } catch (error) {
    console.error('Error sending notification:', error);
  }
};

// Send cry detection notification
export const notifyCryDetection = async (classification) => {
  const cryTypes = {
    hunger: { title: '🍼 Baby is Hungry', body: 'Your baby might need feeding' },
    tired: { title: '😴 Baby is Tired', body: 'Your baby appears tired and may need rest' },
    discomfort: { title: '😣 Baby is Uncomfortable', body: 'Your baby might need a diaper change' },
    belly_pain: { title: '😖 Baby has Belly Pain', body: 'Your baby may be experiencing discomfort' },
    burp: { title: '💨 Baby Needs to Burp', body: 'Your baby needs to burp' },
  };
  
  const notification = cryTypes[classification.predictedClass] || {
    title: '👶 Cry Detected',
    body: 'Your baby is crying',
  };
  
  await sendLocalNotification(
    notification.title,
    `${notification.body} (${(classification.confidence * 100).toFixed(0)}% confidence)`,
    { classification }
  );
};

export default {
  requestNotificationPermissions,
  sendLocalNotification,
  notifyCryDetection,
};
