import AsyncStorage from '@react-native-async-storage/async-storage';
import { Audio } from 'expo-av';

export class SoundManager {
  private static instance: SoundManager;
  private soundEnabled: boolean = true;
  private audioObject: Audio.Sound | null = null;

  private constructor() {}

  static getInstance(): SoundManager {
    if (!SoundManager.instance) {
      SoundManager.instance = new SoundManager();
    }
    return SoundManager.instance;
  }

  async initialize() {
    await this.loadSoundSettings();
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      staysActiveInBackground: true,
      playsInSilentModeIOS: true,
      shouldDuckAndroid: true,
      playThroughEarpieceAndroid: false,
    });
  }

  private async loadSoundSettings() {
    try {
      const sound = await AsyncStorage.getItem('soundEnabled');
      if (sound !== null) {
        this.soundEnabled = JSON.parse(sound);
      }
    } catch (error) {
      console.error('Error loading sound settings:', error);
    }
  }

  async setSoundEnabled(enabled: boolean) {
    this.soundEnabled = enabled;
    await AsyncStorage.setItem('soundEnabled', JSON.stringify(enabled));
  }

  isSoundEnabled(): boolean {
    return this.soundEnabled;
  }

  async playAlertSound(soundFile?: any) {
    if (!this.soundEnabled) return;

    try {
      // Unload previous sound if exists
      if (this.audioObject) {
        await this.audioObject.unloadAsync();
      }

      // Load and play new sound
      if (soundFile) {
        this.audioObject = new Audio.Sound();
        await this.audioObject.loadAsync(soundFile);
        await this.audioObject.playAsync();
      } else {
        // Default notification sound (you can replace with your MP3)
        console.log('Playing default alert sound');
        // await this.audioObject.loadAsync(require('../assets/sounds/alert.mp3'));
        // await this.audioObject.playAsync();
      }
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }

  async cleanup() {
    if (this.audioObject) {
      await this.audioObject.unloadAsync();
      this.audioObject = null;
    }
  }
}

export const soundManager = SoundManager.getInstance();
