import { StyleSheet } from 'react-native';
import { Colors } from '../../constants/theme';
import { useAppColorScheme } from '../../hooks/useAppColorScheme';

export function useAppStyles() {
  const { colorScheme } = useAppColorScheme();
  const isDark = colorScheme === 'dark';

  return StyleSheet.create({
    // ========================
    // General / App-wide Styles
    // ========================
    container: {
      flex: 1,
      backgroundColor: isDark ? Colors.dark.background : Colors.light.background,
      paddingHorizontal: 16,
      paddingTop: 12,
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
      fontSize: 18,
      fontWeight: 'bold',
      color: isDark ? Colors.dark.text : Colors.light.text,
      marginBottom: 12,
      marginTop: 8,
    },
    cardTitle: {
      fontSize: 16,
      fontWeight: '600',
      color: isDark ? Colors.dark.text : Colors.light.text,
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
      backgroundColor: '#60A5FA',
      padding: 14,
      borderRadius: 8,
      alignItems: 'center',
      marginVertical: 20,
    },
    saveButtonText: {
      color: 'white',
      fontSize: 16,
      fontWeight: '600',
    },
    card: {
      backgroundColor: isDark ? '#2a2a2a' : 'white',
      padding: 16,
      borderRadius: 8,
      marginBottom: 12,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: 2,
      elevation: 2,
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
