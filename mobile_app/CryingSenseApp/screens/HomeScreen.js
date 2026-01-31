import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
} from 'react-native';
import CryClassificationCard from '../components/CryClassificationCard';
import { fetchLatestClassifications } from '../services/api';

export default function HomeScreen({ navigation }) {
  const [latestEvent, setLatestEvent] = useState(null);
  const [recentEvents, setRecentEvents] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    // Set up polling for new events every 5 seconds
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const data = await fetchLatestClassifications(5);
      if (data && data.length > 0) {
        setLatestEvent(data[0]);
        setRecentEvents(data.slice(1));
      }
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Latest Cry Detection</Text>
        <TouchableOpacity
          style={styles.settingsButton}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.settingsButtonText}>⚙️</Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <Text>Loading...</Text>
        </View>
      ) : latestEvent ? (
        <CryClassificationCard
          event={latestEvent}
          isLatest={true}
          onPress={() => navigation.navigate('Details', { event: latestEvent })}
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No cry events detected yet</Text>
          <Text style={styles.emptySubtext}>
            Waiting for the baby monitor to detect cries...
          </Text>
        </View>
      )}

      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Events</Text>
          <TouchableOpacity onPress={() => navigation.navigate('History')}>
            <Text style={styles.seeAllText}>See All →</Text>
          </TouchableOpacity>
        </View>

        {recentEvents.length > 0 ? (
          recentEvents.map((event, index) => (
            <CryClassificationCard
              key={event._id || index}
              event={event}
              isLatest={false}
              onPress={() => navigation.navigate('Details', { event })}
            />
          ))
        ) : (
          <Text style={styles.noEventsText}>No recent events</Text>
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  settingsButton: {
    padding: 5,
  },
  settingsButtonText: {
    fontSize: 24,
  },
  loadingContainer: {
    padding: 40,
    alignItems: 'center',
  },
  emptyContainer: {
    padding: 40,
    alignItems: 'center',
    backgroundColor: '#fff',
    margin: 15,
    borderRadius: 10,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#666',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  section: {
    padding: 15,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  seeAllText: {
    fontSize: 14,
    color: '#6200EE',
    fontWeight: '600',
  },
  noEventsText: {
    textAlign: 'center',
    color: '#999',
    padding: 20,
  },
});
