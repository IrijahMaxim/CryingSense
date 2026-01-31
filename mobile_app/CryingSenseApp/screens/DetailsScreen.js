import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import dayjs from 'dayjs';

export default function DetailsScreen({ route }) {
  const { event } = route.params;

  const formatTime = (timestamp) => {
    return dayjs(timestamp).format('MMM D, YYYY h:mm A');
  };

  const getCryTypeDescription = (type) => {
    const descriptions = {
      hunger: 'The baby is likely hungry and needs feeding.',
      tired: 'The baby appears tired and may need rest or sleep.',
      discomfort: 'The baby is uncomfortable and may need a diaper change or repositioning.',
      belly_pain: 'The baby may be experiencing belly pain or gas discomfort.',
      burp: 'The baby needs to burp to relieve trapped air.',
    };
    return descriptions[type] || 'Unknown cry type';
  };

  const getCryTypeEmoji = (type) => {
    const emojis = {
      hunger: '🍼',
      tired: '😴',
      discomfort: '😣',
      belly_pain: '😖',
      burp: '💨',
    };
    return emojis[type] || '👶';
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.card}>
        <View style={styles.header}>
          <Text style={styles.emoji}>{getCryTypeEmoji(event.predictedClass)}</Text>
          <Text style={styles.title}>{event.predictedClass?.replace('_', ' ').toUpperCase()}</Text>
        </View>

        <Text style={styles.description}>{getCryTypeDescription(event.predictedClass)}</Text>

        <View style={styles.infoSection}>
          <Text style={styles.sectionTitle}>Detection Details</Text>
          
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Time:</Text>
            <Text style={styles.infoValue}>{formatTime(event.timestamp)}</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Confidence:</Text>
            <Text style={styles.infoValue}>{(event.confidence * 100).toFixed(1)}%</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Device:</Text>
            <Text style={styles.infoValue}>{event.deviceId}</Text>
          </View>

          {event.inferenceTimeMs && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Processing Time:</Text>
              <Text style={styles.infoValue}>{event.inferenceTimeMs.toFixed(0)} ms</Text>
            </View>
          )}
        </View>

        {event.probabilities && (
          <View style={styles.infoSection}>
            <Text style={styles.sectionTitle}>All Probabilities</Text>
            {Object.entries(event.probabilities).map(([key, value]) => (
              <View key={key} style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>
                  {key.replace('_', ' ')}:
                </Text>
                <View style={styles.probabilityBar}>
                  <View
                    style={[
                      styles.probabilityFill,
                      { width: `${value * 100}%` },
                    ]}
                  />
                </View>
                <Text style={styles.probabilityValue}>
                  {(value * 100).toFixed(1)}%
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  card: {
    backgroundColor: '#fff',
    margin: 15,
    borderRadius: 10,
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  emoji: {
    fontSize: 60,
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
    lineHeight: 24,
  },
  infoSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  probabilityLabel: {
    width: 100,
    fontSize: 12,
    color: '#666',
    textTransform: 'capitalize',
  },
  probabilityBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
    marginHorizontal: 10,
  },
  probabilityFill: {
    height: '100%',
    backgroundColor: '#6200EE',
  },
  probabilityValue: {
    width: 50,
    textAlign: 'right',
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
});
