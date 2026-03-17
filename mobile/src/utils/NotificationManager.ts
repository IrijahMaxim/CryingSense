import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Notifications from 'expo-notifications';

export class NotificationManager {
  private static instance: NotificationManager;
  private notificationsEnabled: boolean = true;

  private constructor() {}

  static getInstance(): NotificationManager {
    if (!NotificationManager.instance) {
      NotificationManager.instance = new NotificationManager();
    }
    return NotificationManager.instance;
  }

  async initialize() {
    await this.loadNotificationSettings();
    
    // Configure notification handler
    Notifications.setNotificationHandler({
      handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: false, // We'll handle sound separately
        shouldSetBadge: false,
        shouldShowBanner: true,
        shouldShowList: true,
      }),
    });
  }

  private async loadNotificationSettings() {
    try {
      const notifications = await AsyncStorage.getItem('notifications');
      if (notifications !== null) {
        this.notificationsEnabled = JSON.parse(notifications);
      }
    } catch (error) {
      console.error('Error loading notification settings:', error);
    }
  }

  async setNotificationsEnabled(enabled: boolean) {
    this.notificationsEnabled = enabled;
    await AsyncStorage.setItem('notifications', JSON.stringify(enabled));
    
    if (!enabled) {
      // Cancel all notifications when disabled
      await Notifications.cancelAllScheduledNotificationsAsync();
    }
  }

  areNotificationsEnabled(): boolean {
    return this.notificationsEnabled;
  }

  async requestPermissions(): Promise<boolean> {
    try {
      const { status } = await Notifications.requestPermissionsAsync();
      return status === 'granted';
    } catch (error) {
      console.error('Error requesting notification permissions:', error);
      return false;
    }
  }

  async scheduleNotification(title: string, body: string, trigger?: Notifications.NotificationTriggerInput) {
    if (!this.notificationsEnabled) return;

    try {
      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          sound: false, // We'll handle sound separately
        },
        trigger: trigger || null,
      });
    } catch (error) {
      console.error('Error scheduling notification:', error);
    }
  }

  async showNotification(title: string, body: string) {
    if (!this.notificationsEnabled) return;

    try {
      // Use scheduleNotificationAsync with immediate trigger for instant notifications
      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          sound: false, // We'll handle sound separately
        },
        trigger: null, // Show immediately
      });
    } catch (error) {
      console.error('Error showing notification:', error);
    }
  }
}

export const notificationManager = NotificationManager.getInstance();
