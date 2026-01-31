"""
MQTT Client for IoT Communication
Handles MQTT communication between ESP32, Raspberry Pi, and Backend
"""

import json
import time


try:
    from umqtt.simple import MQTTClient
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: MQTT library not available - running in simulation mode")


class IoTMQTTClient:
    """
    MQTT client for IoT device communication.
    """
    
    def __init__(self, client_id, broker_host="localhost", broker_port=1883,
                 username=None, password=None, keepalive=60):
        """
        Initialize MQTT client.
        
        Args:
            client_id: Unique client identifier
            broker_host: MQTT broker hostname/IP
            broker_port: MQTT broker port
            username: MQTT username (optional)
            password: MQTT password (optional)
            keepalive: Keep-alive interval in seconds
        """
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.connected = False
        
        # Topic structure
        self.topics = {
            'cry_events': 'cryingsense/cry_events',
            'classifications': 'cryingsense/classifications',
            'device_status': f'cryingsense/devices/{client_id}/status',
            'commands': f'cryingsense/devices/{client_id}/commands'
        }
        
        if MQTT_AVAILABLE:
            self._init_client()
        else:
            print("MQTT client initialized in simulation mode")
    
    def _init_client(self):
        """Initialize MQTT client connection."""
        self.client = MQTTClient(
            self.client_id,
            self.broker_host,
            port=self.broker_port,
            user=self.username,
            password=self.password,
            keepalive=self.keepalive
        )
        
        print(f"MQTT client initialized: {self.client_id}")
        print(f"Broker: {self.broker_host}:{self.broker_port}")
    
    def connect(self):
        """Connect to MQTT broker."""
        if not MQTT_AVAILABLE:
            print("[SIMULATION] Connected to MQTT broker")
            self.connected = True
            return True
        
        try:
            self.client.connect()
            self.connected = True
            print(f"Connected to MQTT broker: {self.broker_host}")
            
            # Publish device status
            self.publish_status("online")
            
            return True
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if not MQTT_AVAILABLE:
            print("[SIMULATION] Disconnected from MQTT broker")
            self.connected = False
            return
        
        try:
            # Publish offline status
            self.publish_status("offline")
            
            self.client.disconnect()
            self.connected = False
            print("Disconnected from MQTT broker")
        except Exception as e:
            print(f"Error disconnecting: {e}")
    
    def publish(self, topic, message, retain=False, qos=0):
        """
        Publish message to topic.
        
        Args:
            topic: MQTT topic
            message: Message to publish (string or dict)
            retain: Retain message flag
            qos: Quality of Service (0, 1, or 2)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MQTT broker")
            return False
        
        try:
            # Convert dict to JSON string
            if isinstance(message, dict):
                message = json.dumps(message)
            
            if not MQTT_AVAILABLE:
                print(f"[SIMULATION] Published to {topic}: {message[:100]}...")
                return True
            
            self.client.publish(topic, message, retain=retain, qos=qos)
            return True
        
        except Exception as e:
            print(f"Failed to publish message: {e}")
            return False
    
    def publish_cry_event(self, event_data):
        """
        Publish cry detection event.
        
        Args:
            event_data: Dictionary containing cry event data
        
        Returns:
            True if successful, False otherwise
        """
        return self.publish(self.topics['cry_events'], event_data)
    
    def publish_classification(self, classification_data):
        """
        Publish cry classification result.
        
        Args:
            classification_data: Dictionary containing classification result
        
        Returns:
            True if successful, False otherwise
        """
        return self.publish(self.topics['classifications'], classification_data)
    
    def publish_status(self, status="online"):
        """
        Publish device status.
        
        Args:
            status: Device status (online, offline, error, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        status_data = {
            'device_id': self.client_id,
            'status': status,
            'timestamp': time.time()
        }
        
        return self.publish(self.topics['device_status'], status_data, retain=True)
    
    def subscribe(self, topic, callback=None):
        """
        Subscribe to topic.
        
        Args:
            topic: MQTT topic to subscribe to
            callback: Callback function for received messages
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MQTT broker")
            return False
        
        try:
            if not MQTT_AVAILABLE:
                print(f"[SIMULATION] Subscribed to {topic}")
                return True
            
            self.client.set_callback(callback if callback else self._default_callback)
            self.client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")
            return True
        
        except Exception as e:
            print(f"Failed to subscribe: {e}")
            return False
    
    def _default_callback(self, topic, message):
        """Default callback for received messages."""
        print(f"Received message on {topic}: {message}")
    
    def check_messages(self):
        """Check for new messages (non-blocking)."""
        if not self.connected or not MQTT_AVAILABLE:
            return
        
        try:
            self.client.check_msg()
        except Exception as e:
            print(f"Error checking messages: {e}")
    
    def wait_for_messages(self, timeout=None):
        """Wait for messages (blocking)."""
        if not self.connected or not MQTT_AVAILABLE:
            return
        
        try:
            self.client.wait_msg()
        except Exception as e:
            print(f"Error waiting for messages: {e}")


def test_mqtt_client():
    """Test MQTT client functionality."""
    print("="*60)
    print("Testing MQTT Client")
    print("="*60)
    
    # Initialize client
    client = IoTMQTTClient(
        client_id="TEST_ESP32",
        broker_host="localhost",
        broker_port=1883
    )
    
    # Connect
    print("\nConnecting to broker...")
    client.connect()
    
    # Publish cry event
    print("\nPublishing cry event...")
    cry_event = {
        'device_id': 'TEST_ESP32',
        'timestamp': time.time(),
        'features': {
            'rms_energy': 0.45,
            'zcr': 0.12,
            'mean_amplitude': 0.38
        }
    }
    client.publish_cry_event(cry_event)
    
    # Publish classification
    print("\nPublishing classification...")
    classification = {
        'device_id': 'TEST_ESP32',
        'timestamp': time.time(),
        'predicted_class': 'hunger',
        'confidence': 0.87
    }
    client.publish_classification(classification)
    
    # Wait a bit
    time.sleep(1)
    
    # Disconnect
    print("\nDisconnecting...")
    client.disconnect()
    
    print("="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_mqtt_client()
