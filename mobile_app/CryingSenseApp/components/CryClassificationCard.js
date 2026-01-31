import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);

export default function CryClassificationCard({ event, isLatest, onPress }) {
  const getCryTypeColor = (type) => {
    const colors = {
      hunger: '#FF9800',
      tired: '#2196F3',
      discomfort: '#F44336',
      belly_pain: '#E91E63',
      burp: '#4CAF50',
    };
    return colors[type] || '#9E9E9E';
  };

  const getCryTypeEmoji = (type) => {
    const emojis = {
      hunger: '🍼',
      tired: '😴',
      discomfort: '😣',
      belly_pain: '😖',
      burp: '��',
    };
    return emojis[type] || '👶';
  };

  const formatTimeAgo = (timestamp) => {
    return dayjs(timestamp).fromNow();
  };

  return (
    <TouchableOpacity
      style={[
        styles.card,
        isLatest && styles.latestCard,
        { borderLeftColor: getCryTypeColor(event.predictedClass) },
      ]}
      onPress={onPress}
    >
      <View style={styles.content}>
        <View style={styles.leftContent}>
          <Text style={styles.emoji}>{getCryTypeEmoji(event.predictedClass)}</Text>
        </View>
        
        <View style={styles.middleContent}>
          <Text style={styles.cryType}>
            {event.predictedClass?.replace('_', ' ').toUpperCase()}
          </Text>
          <Text style={styles.timestamp}>{formatTimeAgo(event.timestamp)}</Text>
          <Text style={styles.device}>{event.deviceId}</Text>
        </View>
        
        <View style={styles.rightContent}>
          <Text style={styles.confidence}>
            {(event.confidence * 100).toFixed(0)}%
          </Text>
          <Text style={styles.confidenceLabel}>confidence</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginVertical: 5,
    marginHorizontal: 10,
    borderLeftWidth: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  latestCard: {
    borderWidth: 2,
    borderColor: '#6200EE',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  leftContent: {
    marginRight: 15,
  },
  emoji: {
    fontSize: 40,
  },
  middleContent: {
    flex: 1,
  },
  cryType: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  timestamp: {
    fontSize: 12,
    color: '#999',
    marginBottom: 2,
  },
  device: {
    fontSize: 11,
    color: '#666',
  },
  rightContent: {
    alignItems: 'flex-end',
  },
  confidence: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6200EE',
  },
  confidenceLabel: {
    fontSize: 10,
    color: '#999',
  },
});
