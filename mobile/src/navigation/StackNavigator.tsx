// navigation/StackNavigator.tsx (or App.tsx if you define it there)
import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import SettingsScreen from '../screens/SettingsScreen';
import EditBabyProfileScreen from '../screens/EditBabyProfileScreen';
import AnalysisResultScreen from '../screens/AnalysisResultScreen';
import HistoryAndInsightsScreen from '../screens/HistoryScreen';

export type RootStackParamList = {
  HomeScreen: undefined;
  SettingsScreen: undefined;
  AnalysisResultScreen: undefined;
  HistoryScreen: undefined;
  EditBabyProfileScreen: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppStack() {
  return (
    <Stack.Navigator initialRouteName="HomeScreen">
      <Stack.Screen name="HomeScreen" component={HomeScreen} />
      <Stack.Screen name="SettingsScreen" component={SettingsScreen} />
      <Stack.Screen name="EditBabyProfileScreen" component={EditBabyProfileScreen} />
      <Stack.Screen name="AnalysisResultScreen" component={AnalysisResultScreen} />
      <Stack.Screen name="HistoryScreen" component={HistoryAndInsightsScreen} />
    </Stack.Navigator>
  );
}
