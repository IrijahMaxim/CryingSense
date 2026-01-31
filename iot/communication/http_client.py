"""
HTTP Client for IoT Communication
Handles HTTP/REST API communication with backend server
"""

import json
import time


try:
    import urequests as requests
    HTTP_AVAILABLE = True
except ImportError:
    try:
        import requests
        HTTP_AVAILABLE = True
    except ImportError:
        HTTP_AVAILABLE = False
        print("Warning: HTTP library not available - running in simulation mode")


class HTTPClient:
    """
    HTTP client for REST API communication.
    """
    
    def __init__(self, base_url="http://localhost:3000", timeout=10):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL of the backend server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        print(f"HTTP client initialized")
        print(f"Base URL: {self.base_url}")
    
    def post(self, endpoint, data, headers=None):
        """
        Send POST request.
        
        Args:
            endpoint: API endpoint (e.g., '/api/cry-event')
            data: Data to send (dict or string)
            headers: Optional headers dict
        
        Returns:
            Response dict or None on error
        """
        url = f"{self.base_url}{endpoint}"
        
        # Default headers
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        # Convert data to JSON if needed
        if isinstance(data, dict):
            data = json.dumps(data)
        
        if not HTTP_AVAILABLE:
            print(f"[SIMULATION] POST {url}")
            print(f"[SIMULATION] Data: {data[:100]}...")
            return {'status': 'success', 'message': 'Simulated response'}
        
        try:
            response = requests.post(url, data=data, headers=headers, timeout=self.timeout)
            
            # Parse response
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    return {'status': 'success', 'message': response.text}
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"HTTP request failed: {e}")
            return None
    
    def get(self, endpoint, params=None, headers=None):
        """
        Send GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters (dict)
            headers: Optional headers dict
        
        Returns:
            Response dict or None on error
        """
        url = f"{self.base_url}{endpoint}"
        
        if not HTTP_AVAILABLE:
            print(f"[SIMULATION] GET {url}")
            return {'status': 'success', 'data': []}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    return {'status': 'success', 'data': response.text}
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"HTTP request failed: {e}")
            return None
    
    def send_cry_event(self, event_data):
        """
        Send cry detection event to backend.
        
        Args:
            event_data: Dictionary containing cry event data
        
        Returns:
            Response dict or None on error
        """
        return self.post('/api/cry-event', event_data)
    
    def send_classification(self, classification_data):
        """
        Send classification result to backend.
        
        Args:
            classification_data: Dictionary containing classification result
        
        Returns:
            Response dict or None on error
        """
        return self.post('/api/classification', classification_data)
    
    def send_device_status(self, device_id, status="online"):
        """
        Send device status update.
        
        Args:
            device_id: Device identifier
            status: Device status
        
        Returns:
            Response dict or None on error
        """
        status_data = {
            'device_id': device_id,
            'status': status,
            'timestamp': time.time()
        }
        
        return self.post('/api/device-status', status_data)
    
    def get_device_config(self, device_id):
        """
        Get device configuration from backend.
        
        Args:
            device_id: Device identifier
        
        Returns:
            Configuration dict or None on error
        """
        return self.get(f'/api/device-config/{device_id}')
    
    def send_batch_events(self, events):
        """
        Send multiple events in a single request.
        
        Args:
            events: List of event dictionaries
        
        Returns:
            Response dict or None on error
        """
        batch_data = {
            'events': events,
            'count': len(events),
            'timestamp': time.time()
        }
        
        return self.post('/api/batch-events', batch_data)


def test_http_client():
    """Test HTTP client functionality."""
    print("="*60)
    print("Testing HTTP Client")
    print("="*60)
    
    # Initialize client
    client = HTTPClient(base_url="http://localhost:3000")
    
    # Test cry event
    print("\nSending cry event...")
    cry_event = {
        'device_id': 'ESP32_001',
        'timestamp': time.time(),
        'features': {
            'rms_energy': 0.45,
            'zcr': 0.12,
            'mean_amplitude': 0.38,
            'spectral_centroid': 3200,
            'duration': 5.0,
            'num_samples': 80000
        },
        'envelope': [0.5, 0.6, 0.7, 0.8, 0.9]
    }
    
    response = client.send_cry_event(cry_event)
    print(f"Response: {response}")
    
    # Test classification
    print("\nSending classification...")
    classification = {
        'device_id': 'ESP32_001',
        'timestamp': time.time(),
        'predicted_class': 'hunger',
        'confidence': 0.87,
        'probabilities': {
            'belly_pain': 0.05,
            'burp': 0.03,
            'discomfort': 0.02,
            'hunger': 0.87,
            'tired': 0.03
        },
        'inference_time_ms': 125.4
    }
    
    response = client.send_classification(classification)
    print(f"Response: {response}")
    
    # Test device status
    print("\nSending device status...")
    response = client.send_device_status('ESP32_001', 'online')
    print(f"Response: {response}")
    
    # Test get config
    print("\nGetting device config...")
    config = client.get_device_config('ESP32_001')
    print(f"Config: {config}")
    
    print("="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_http_client()
