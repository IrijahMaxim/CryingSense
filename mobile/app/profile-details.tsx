import React from 'react';
import { View, Text, ScrollView, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useBaby } from '../src/context/BabyContext';
import { useTheme } from '../src/context/ThemeContext';

export default function ProfileDetailsScreen() {
  const { profile } = useBaby();
  const { colors, isDark } = useTheme();

  return (
    <ScrollView style={{ flex: 1, backgroundColor: colors.background, paddingHorizontal: 16, paddingTop: 12 }}>
      {/* Header */}
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={28} color={colors.text} />
        </TouchableOpacity>
        <Text style={{ fontSize: 18, fontWeight: 'bold', color: colors.text }}>Baby Profile</Text>
        <TouchableOpacity onPress={() => router.push('/edit-baby-profile')}>
          <Ionicons name="create-outline" size={28} color={colors.text} />
        </TouchableOpacity>
      </View>

      {/* Profile Picture */}
      <View style={{ 
        backgroundColor: isDark ? '#2a2a2a' : 'white', 
        padding: 24, 
        borderRadius: 16, 
        alignItems: 'center', 
        marginBottom: 20, 
        shadowColor: '#000', 
        shadowOffset: { width: 0, height: 4 }, 
        shadowOpacity: isDark ? 0.3 : 0.1, 
        shadowRadius: 8, 
        elevation: 5 
      }}>
        {profile?.photo ? (
          <Image
            source={{ uri: profile.photo }}
            style={{ width: 120, height: 120, borderRadius: 60, marginBottom: 16 }}
          />
        ) : (
          <Image
            source={require('../assets/baby_placeholder.png')}
            style={{ width: 120, height: 120, borderRadius: 60, marginBottom: 16 }}
          />
        )}
        <Text style={{ fontSize: 24, fontWeight: 'bold', color: colors.text, marginBottom: 8 }}>
          {profile?.name || 'Baby'}
        </Text>
        <Text style={{ fontSize: 18, color: isDark ? '#999' : '#666' }}>
          {profile?.ageMonths || 0} months old
        </Text>
      </View>

      {/* Profile Information */}
      <View style={{ 
        backgroundColor: isDark ? '#2a2a2a' : 'white', 
        padding: 20, 
        borderRadius: 12, 
        marginBottom: 16, 
        shadowColor: '#000', 
        shadowOffset: { width: 0, height: 2 }, 
        shadowOpacity: isDark ? 0.3 : 0.1, 
        shadowRadius: 4, 
        elevation: 3 
      }}>
        <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 12 }}>Profile Information</Text>
        
        <View style={{ marginBottom: 12 }}>
          <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 4 }}>Name</Text>
          <Text style={{ fontSize: 16, color: colors.text }}>{profile?.name || 'Baby'}</Text>
        </View>
        
        <View style={{ marginBottom: 12 }}>
          <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 4 }}>Age</Text>
          <Text style={{ fontSize: 16, color: colors.text }}>{profile?.ageMonths || 0} months</Text>
        </View>
        
        <View>
          <Text style={{ fontSize: 14, color: isDark ? '#999' : '#666', marginBottom: 4 }}>Profile Status</Text>
          <View style={{ 
            backgroundColor: '#4CAF50', 
            paddingHorizontal: 12, 
            paddingVertical: 6, 
            borderRadius: 8, 
            alignSelf: 'flex-start',
            marginTop: 4 
          }}>
            <Text style={{ fontSize: 14, color: 'white', fontWeight: '600' }}>Active</Text>
          </View>
        </View>
      </View>

      {/* Quick Actions */}
      <View style={{ 
        backgroundColor: isDark ? '#2a2a2a' : 'white', 
        padding: 20, 
        borderRadius: 12, 
        shadowColor: '#000', 
        shadowOffset: { width: 0, height: 2 }, 
        shadowOpacity: isDark ? 0.3 : 0.1, 
        shadowRadius: 4, 
        elevation: 3 
      }}>
        <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 16 }}>Quick Actions</Text>
        
        <TouchableOpacity 
          style={{ 
            backgroundColor: '#60A5FA', 
            padding: 14, 
            borderRadius: 8, 
            alignItems: 'center', 
            marginBottom: 12,
            flexDirection: 'row',
            justifyContent: 'center'
          }}
          onPress={() => router.push('/edit-baby-profile')}
        >
          <Ionicons name="create-outline" size={20} color="white" style={{ marginRight: 8 }} />
          <Text style={{ fontSize: 16, fontWeight: '600', color: 'white' }}>Edit Profile</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={{ 
            backgroundColor: isDark ? '#333' : '#f5f5f5', 
            padding: 14, 
            borderRadius: 8, 
            alignItems: 'center',
            borderWidth: 1,
            borderColor: isDark ? '#555' : '#ddd'
          }}
          onPress={() => router.push('/settings')}
        >
          <Ionicons name="settings-outline" size={20} color={colors.text} style={{ marginRight: 8 }} />
          <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>Settings</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}
