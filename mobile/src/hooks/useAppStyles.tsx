import { StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { Colors } from '../constants/theme';

export function useAppStyles() {
  const { isDark } = useTheme();

  return StyleSheet.create({
    // ========================
    // General / App-wide Styles
    // ========================
    container: {
      flex: 1,
      backgroundColor: isDark ? Colors.dark.background : Colors.light.background,
      paddingHorizontal: 0,
      paddingTop: 0,
      marginBottom: 20,
    },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingHorizontal: 0,
      paddingTop: 0,
      marginBottom: 20,
    },
    sectionTitle: {
      fontSize: 24,
      fontWeight: 'bold',
      color: isDark ? Colors.light.text : Colors.dark.text,
      marginBottom: 16,
    },
    cardTitle: {
      fontSize: 18,
      fontWeight: '600',
      color: isDark ? Colors.light.text : Colors.dark.text,
      marginBottom: 8,
    },
    textInput: {
      backgroundColor: isDark ? '#2a2a2a' : 'white',
      color: isDark ? Colors.dark.text : Colors.light.text,
      padding: 12,
      borderRadius: 8,
      borderWidth: 1,
      borderColor: isDark ? '#444' : '#ddd',
      fontSize: 14,
      marginBottom: 12,
    },
    saveButton: {
      backgroundColor: Colors.light.primary,
      paddingHorizontal: 16,
      paddingVertical: 12,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
      marginVertical: 20,
    },
    saveButtonText: {
      color: '#fff',
      fontSize: 16,
      fontWeight: '600',
    },
    card: {
      backgroundColor: isDark ? Colors.dark.surface : Colors.light.surface,
      borderRadius: 12,
      padding: 16,
      marginBottom: 16,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    preferenceRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: 12,
      backgroundColor: isDark ? '#2a2a2a' : 'white',
      borderRadius: 8,
      marginBottom: 8,
      paddingHorizontal: 16,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    preferenceText: {
      fontSize: 16,
      color: isDark ? Colors.dark.text : Colors.light.text,
    },
    insightTitle: {
      fontSize: 16,
      fontWeight: '600',
      color: isDark ? Colors.dark.text : Colors.light.text,
      marginBottom: 8,
    },

    // ========================
    // History & Insights Screen
    // ========================
    historyCard: {
      backgroundColor: isDark ? '#2a2a2a' : 'white',
      padding: 12,
      borderRadius: 8,
      marginBottom: 10,
      flexDirection: 'row',
      alignItems: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    historyText: {
      fontSize: 16,
      color: isDark ? Colors.dark.text : Colors.light.text,
    },
    historyTime: {
      fontSize: 14,
      color: isDark ? '#999' : '#999',
    },

    // ========================
    // Edit Baby Profile Screen
    // ========================
    avatarCircle: {
      width: 50,
      height: 50,
      borderRadius: 25,
      borderWidth: 2,
      borderColor: isDark ? '#555' : '#ddd',
      backgroundColor: isDark ? '#333' : '#f9f9f9',
    },
  });
}
