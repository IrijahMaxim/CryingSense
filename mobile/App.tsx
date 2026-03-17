import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import BottomTabs from './src/navigation/BottomTabs';
import AnalysisResultScreen from './src/screens/AnalysisResultScreen';
import EditBabyProfileScreen from './src/screens/EditBabyProfileScreen';
import { BabyProvider } from './src/context/BabyContext';

export type RootStackParamList = {
  MainTabs: undefined;
  AnalysisResultScreen: undefined;
  EditBabyProfileScreen: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <BabyProvider>
      <NavigationContainer>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          <Stack.Screen name="MainTabs" component={BottomTabs} />
          <Stack.Screen name="AnalysisResultScreen" component={AnalysisResultScreen} />
          <Stack.Screen name="EditBabyProfileScreen" component={EditBabyProfileScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </BabyProvider>
  );
}
