#include <Arduino.h>
// #include <WiFi.h>        // COMMENTED OUT - Using COM port instead
// #include <WiFiUdp.h>     // COMMENTED OUT - Using COM port instead
#include <driver/i2s.h>

/*
 * ESP32 INMP441 Audio Transmitter over COM Port (Serial)
 * 
 * Captures audio from INMP441 microphone and sends to PC/Raspberry Pi
 * via Serial (USB) for real-time cry classification.
 * 
 * HARDWARE CONNECTIONS:
 *   INMP441     ESP32
 *   VDD    ->   3.3V
 *   GND    ->   GND
 *   SD     ->   D16
 *   WS     ->   D17
 *   SCK    ->   D18
 *   L/R    ->   GND (left channel)
 * 
 * HARDWARE TIPS TO REDUCE NOISE/SPIKES:
 * 1. Use a stable 3.3V power supply (not USB 5V with voltage regulator if possible)
 * 2. Add 10uF capacitor between VDD and GND on INMP441 (close to mic)
 * 3. Add 100nF ceramic capacitor between VDD and GND on ESP32
 * 4. Keep I2S wires short and away from power lines
 * 5. Use common ground between ESP32 and INMP441
 * 6. Twist I2S data wires together to reduce interference
 */

// =============================================================================
// WIFI CONFIGURATION - COMMENTED OUT (Using COM port)
// =============================================================================
// const char* WIFI_SSID = "DESKTOP-3DNLEM0 8867";      // Your WiFi network name
// const char* WIFI_PASSWORD = "8292J?a2";   // Your WiFi password

// Raspberry Pi / Computer IP and port - COMMENTED OUT (Using COM port)
// const char* SERVER_IP = "192.168.1.9";      // IP of computer running main.py
// const int SERVER_PORT = 8888;                  // Must match config.py WIFI_PORT

// =============================================================================
// SERIAL CONFIGURATION
// =============================================================================
#define SERIAL_BAUD_RATE 115200
#define SERIAL_SYNC_BYTE_1 0xAA
#define SERIAL_SYNC_BYTE_2 0x55

// =============================================================================
// I2S MICROPHONE CONFIGURATION
// =============================================================================
#define I2S_WS    17   // Word Select (LRCLK)
#define I2S_SD    16   // Serial Data (DOUT)
#define I2S_SCK   18   // Serial Clock (BCLK)
#define I2S_PORT  I2S_NUM_0

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 512
#define BYTES_TO_READ (BUFFER_SIZE * 4)

// =============================================================================
// LED CONFIGURATION
// =============================================================================
#define LED_PIN 2
#define LED_PWM_CHANNEL 0
#define LED_PWM_FREQ 5000
#define LED_PWM_RESOLUTION 8

// =============================================================================
// AUDIO PROCESSING
// =============================================================================
#define SOFTWARE_GAIN 1.0              // Amplification factor (increase if needed)

// Baby Crying Detection thresholds
#define AMBIENT_THRESHOLD 30.0         // Normal room noise level
#define CRYING_THRESHOLD 100.0         // Baby crying detection threshold
#define LOUD_CRY_THRESHOLD 250.0       // Very loud crying
#define MAX_AMPLITUDE 600.0            // Maximum expected amplitude

// Sustained sound detection (to filter brief noises)
#define DETECTION_COUNT 2              // Consecutive detections needed
int loudCount = 0;                     // Counter for sustained detection

// =============================================================================
// PACKET PROTOCOL
// =============================================================================
// Header format: [packet_id (4B), timestamp_ms (4B), sample_count (2B), flags (2B)]
#define HEADER_SIZE 12
#define FLAG_FIRST_PACKET 0x01
#define FLAG_LAST_PACKET 0x02
#define FLAG_CRY_DETECTED 0x04

// =============================================================================
// GLOBALS
// =============================================================================
// WiFiUDP udp;                              // COMMENTED OUT - Using COM port
uint32_t packetId = 0;
bool firstPacket = true;
// unsigned long lastWiFiCheck = 0;          // COMMENTED OUT - Using COM port
// const unsigned long WIFI_RETRY_INTERVAL = 10000;  // COMMENTED OUT

int16_t sBuffer[BUFFER_SIZE];
uint8_t txBuffer[HEADER_SIZE + BUFFER_SIZE * 2];  // Header + 16-bit samples

// Forward declarations
void i2s_install();
void i2s_setpin();
// void connectWiFi();                       // COMMENTED OUT - Using COM port
// bool sendAudioPacket(int16_t* samples, int count, uint16_t flags);  // COMMENTED OUT
bool sendAudioPacketSerial(int16_t* samples, int count, uint16_t flags);  // Serial version

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("CryingSense ESP32 Audio Transmitter");
    Serial.println("Mode: COM Port (Serial)");
    Serial.println("========================================");
    
    // Setup LED
    ledcSetup(LED_PWM_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
    ledcAttachPin(LED_PIN, LED_PWM_CHANNEL);
    ledcWrite(LED_PWM_CHANNEL, 0);
    Serial.println("LED configured");
    
    // Connect to WiFi - COMMENTED OUT (Using COM port)
    // connectWiFi();
    
    // Initialize I2S
    Serial.println("Initializing I2S...");
    i2s_install();
    i2s_setpin();
    i2s_start(I2S_PORT);
    
    // Clear initial buffer
    size_t bytes_read;
    int32_t dummy[BUFFER_SIZE];
    for (int i = 0; i < 5; i++) {
        i2s_read(I2S_PORT, &dummy, BYTES_TO_READ, &bytes_read, 100);
        delay(10);
    }
    Serial.println("I2S initialized and buffer cleared");
    
    // Initialize UDP - COMMENTED OUT (Using COM port)
    // udp.begin(8889);  // Local port (different from server port)
    // Serial.println("UDP initialized");
    
    Serial.println("\nReady to transmit audio via Serial!");
    Serial.print("Baud Rate: ");
    Serial.println(SERIAL_BAUD_RATE);
    Serial.print("Software Gain: ");
    Serial.print(SOFTWARE_GAIN);
    Serial.println("x");
    Serial.println("Thresholds - Ambient: 30 | Crying: 100 | Loud: 250");
    Serial.println("========================================\n");
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
    // Periodically check and retry WiFi connection (non-blocking) - COMMENTED OUT
    // unsigned long currentTime = millis();
    // if (WiFi.status() != WL_CONNECTED && (currentTime - lastWiFiCheck > WIFI_RETRY_INTERVAL)) {
    //     Serial.println("WiFi disconnected! Attempting reconnect...");
    //     connectWiFi();
    //     lastWiFiCheck = currentTime;
    // }
    
    // Read audio from I2S
    size_t bytesIn = 0;
    int32_t raw32Buffer[BUFFER_SIZE];
    
    esp_err_t result = i2s_read(I2S_PORT, &raw32Buffer, BYTES_TO_READ, &bytesIn, portMAX_DELAY);
    
    if (result != ESP_OK || bytesIn == 0) {
        return;
    }
    
    int samples_read = bytesIn / 4;
    
    // Convert 32-bit to 16-bit
    for (int i = 0; i < samples_read; ++i) {
        sBuffer[i] = (int16_t)(raw32Buffer[i] >> 16);
    }
    
    // Remove DC offset
    long dc_sum = 0;
    for (int i = 0; i < samples_read; ++i) {
        dc_sum += sBuffer[i];
    }
    int16_t dc_offset = dc_sum / samples_read;
    
    for (int i = 0; i < samples_read; ++i) {
        sBuffer[i] = sBuffer[i] - dc_offset;
    }
    
    // Apply software gain for increased sensitivity
    for (int i = 0; i < samples_read; ++i) {
        float amplified = (float)sBuffer[i] * SOFTWARE_GAIN;
        amplified = constrain(amplified, -32768, 32767);
        sBuffer[i] = (int16_t)amplified;
    }
    
    // Calculate amplitude with spike rejection filter
    long sum_abs = 0;
    int16_t peak = 0;
    int valid_samples = 0;
    
    // First pass: calculate rough average to detect spikes
    long rough_sum = 0;
    for (int i = 0; i < samples_read; ++i) {
        rough_sum += abs(sBuffer[i]);
    }
    int rough_avg = rough_sum / samples_read;
    int spike_threshold = rough_avg * 10; // Reject samples > 10x average
    
    // Second pass: calculate amplitude excluding spikes
    for (int i = 0; i < samples_read; ++i) {
        int16_t abs_val = abs(sBuffer[i]);
        
        // Reject obvious spikes (likely power noise)
        if (abs_val < spike_threshold || spike_threshold == 0) {
            sum_abs += abs_val;
            valid_samples++;
            if (abs_val > peak) peak = abs_val;
        }
    }
    
    int amplitude = valid_samples > 0 ? sum_abs / valid_samples : 0;
    
    // Update LED brightness based on amplitude
    int brightness = 0;
    if (amplitude > AMBIENT_THRESHOLD) {
        brightness = (int)((amplitude / MAX_AMPLITUDE) * 255.0);
        brightness = constrain(brightness, 0, 255);
    }
    ledcWrite(LED_PWM_CHANNEL, brightness);
    
    // Baby cry detection with sustained sound check
    if (amplitude > CRYING_THRESHOLD) {
        loudCount++;
        if (loudCount >= DETECTION_COUNT) {
            // Mark cry detected in flags for next packet
            loudCount = DETECTION_COUNT; // Cap the counter
        }
    } else {
        // Reset counter if sound drops below threshold
        if (loudCount > 0) {
            loudCount--;
        }
    }
    
    // Set flags
    uint16_t flags = 0;
    if (firstPacket) {
        flags |= FLAG_FIRST_PACKET;
        firstPacket = false;
    }
    if (loudCount >= DETECTION_COUNT) {
        flags |= FLAG_CRY_DETECTED;
    }
    
    // Send audio packet (only if WiFi connected) - COMMENTED OUT
    // if (WiFi.status() == WL_CONNECTED) {
    //     sendAudioPacket(sBuffer, samples_read, flags);
    // }
    
    // Send audio packet via Serial (COM port)
    sendAudioPacketSerial(sBuffer, samples_read, flags);
    
    // Periodic status output
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 2000) {
        Serial.print("Amp: ");
        Serial.print(amplitude);
        Serial.print(" | Peak: ");
        Serial.print(peak);
        Serial.print(" | LED: ");
        Serial.print(brightness);
        Serial.print(" | Packets: ");
        Serial.print(packetId);
        Serial.print(" | Status: ");
        if (amplitude < AMBIENT_THRESHOLD) {
            Serial.print("Quiet");
        } else if (amplitude < CRYING_THRESHOLD) {
            Serial.print("Normal");
        } else if (amplitude < LOUD_CRY_THRESHOLD) {
            Serial.print("CRYING");
        } else {
            Serial.print("LOUD CRY!");
        }
        Serial.println(" | Mode: Serial");
        // WiFi status output - COMMENTED OUT
        // if (WiFi.status() == WL_CONNECTED) {
        //     Serial.print(" | RSSI: ");
        //     Serial.print(WiFi.RSSI());
        //     Serial.print(" dBm");
        // } else {
        //     Serial.print(" | WiFi: OFFLINE");
        // }
        // Serial.println();
        lastStatus = millis();
    }
}

// =============================================================================
// WIFI CONNECTION - COMMENTED OUT (Using COM port)
// =============================================================================
/*
void connectWiFi() {
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
        
        // Blink LED while connecting
        ledcWrite(LED_PWM_CHANNEL, (attempts % 2) ? 128 : 0);
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println(" Connected!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        Serial.print("MAC Address: ");
        Serial.println(WiFi.macAddress());
        ledcWrite(LED_PWM_CHANNEL, 0);
    } else {
        Serial.println(" Failed!");
        Serial.println("Continuing in offline mode - audio will be captured but not sent");
        Serial.println("Will retry WiFi connection in loop...");
        ledcWrite(LED_PWM_CHANNEL, 0);
    }
}
*/

// =============================================================================
// SEND AUDIO PACKET VIA SERIAL (COM PORT)
// =============================================================================
bool sendAudioPacketSerial(int16_t* samples, int count, uint16_t flags) {
    // Build header
    uint32_t timestamp = millis();
    
    // Send sync bytes first (to identify packet start)
    Serial.write(SERIAL_SYNC_BYTE_1);
    Serial.write(SERIAL_SYNC_BYTE_2);
    
    // Send header: packet_id (4B), timestamp (4B), sample_count (2B), flags (2B)
    Serial.write((uint8_t*)&packetId, 4);
    Serial.write((uint8_t*)&timestamp, 4);
    Serial.write((uint8_t*)&count, 2);
    Serial.write((uint8_t*)&flags, 2);
    
    // Send audio samples (16-bit, little-endian)
    Serial.write((uint8_t*)samples, count * 2);
    
    packetId++;
    
    return true;
}

// =============================================================================
// SEND AUDIO PACKET VIA UDP - COMMENTED OUT (Using COM port)
// =============================================================================
/*
bool sendAudioPacket(int16_t* samples, int count, uint16_t flags) {
    // Safety check - don't send if WiFi not connected
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }
    
    // Build header
    uint32_t timestamp = millis();
    
    // Pack header: packet_id (4B), timestamp (4B), sample_count (2B), flags (2B)
    memcpy(txBuffer, &packetId, 4);
    memcpy(txBuffer + 4, &timestamp, 4);
    memcpy(txBuffer + 8, &count, 2);
    memcpy(txBuffer + 10, &flags, 2);
    
    // Copy audio samples (16-bit, little-endian)
    memcpy(txBuffer + HEADER_SIZE, samples, count * 2);
    
    // Send UDP packet
    int totalSize = HEADER_SIZE + (count * 2);
    
    udp.beginPacket(SERVER_IP, SERVER_PORT);
    size_t written = udp.write(txBuffer, totalSize);
    int result = udp.endPacket();
    
    packetId++;
    
    return (result == 1 && written == totalSize);
}
*/

// =============================================================================
// I2S CONFIGURATION
// =============================================================================
void i2s_install() {
    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = i2s_bits_per_sample_t(32),
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = false
    };
    
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    
    i2s_set_pin(I2S_PORT, &pin_config);
}
