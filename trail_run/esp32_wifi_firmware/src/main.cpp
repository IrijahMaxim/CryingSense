#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

/*
 * ESP32 INMP441 Audio Transmitter over WiFi
 * 
 * Captures audio from INMP441 microphone and sends to Raspberry Pi
 * via UDP for real-time cry classification.
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
 * NOISE REDUCTION TIPS:
 * 1. Use stable 3.3V power supply
 * 2. Add 10uF capacitor between VDD and GND on INMP441
 * 3. Keep I2S wires short and twisted together
 */

// =============================================================================
// WIFI CONFIGURATION - CHANGE THESE!
// =============================================================================
const char* WIFI_SSID = "YourWiFiName";      // Your WiFi network name
const char* WIFI_PASSWORD = "YourPassword";   // Your WiFi password

// Raspberry Pi / Computer IP and port - CHANGE THIS!
const char* SERVER_IP = "192.168.1.100";      // IP of computer running main.py
const int SERVER_PORT = 8888;                  // Must match config.py WIFI_PORT

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
#define SOFTWARE_GAIN 1.0
#define AMPLITUDE_THRESHOLD 100   // Minimum amplitude for LED response

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
WiFiUDP udp;
uint32_t packetId = 0;
bool firstPacket = true;

int16_t sBuffer[BUFFER_SIZE];
uint8_t txBuffer[HEADER_SIZE + BUFFER_SIZE * 2];  // Header + 16-bit samples

// Forward declarations
void i2s_install();
void i2s_setpin();
void connectWiFi();
bool sendAudioPacket(int16_t* samples, int count, uint16_t flags);

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("CryingSense ESP32 Audio Transmitter");
    Serial.println("========================================");
    
    // Setup LED
    ledcSetup(LED_PWM_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
    ledcAttachPin(LED_PIN, LED_PWM_CHANNEL);
    ledcWrite(LED_PWM_CHANNEL, 0);
    Serial.println("LED configured");
    
    // Connect to WiFi
    connectWiFi();
    
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
    
    // Initialize UDP
    udp.begin(8889);  // Local port (different from server port)
    Serial.println("UDP initialized");
    
    Serial.println("\nReady to transmit audio!");
    Serial.print("Sending to: ");
    Serial.print(SERVER_IP);
    Serial.print(":");
    Serial.println(SERVER_PORT);
    Serial.println("========================================\n");
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected! Reconnecting...");
        ledcWrite(LED_PWM_CHANNEL, 255);  // Full brightness = error
        connectWiFi();
        ledcWrite(LED_PWM_CHANNEL, 0);
    }
    
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
    
    // Apply software gain
    if (SOFTWARE_GAIN != 1.0) {
        for (int i = 0; i < samples_read; ++i) {
            float amplified = (float)sBuffer[i] * SOFTWARE_GAIN;
            amplified = constrain(amplified, -32768, 32767);
            sBuffer[i] = (int16_t)amplified;
        }
    }
    
    // Calculate amplitude for LED
    long sum_abs = 0;
    for (int i = 0; i < samples_read; ++i) {
        sum_abs += abs(sBuffer[i]);
    }
    int amplitude = sum_abs / samples_read;
    
    // Update LED brightness based on amplitude
    int brightness = 0;
    if (amplitude > AMPLITUDE_THRESHOLD) {
        brightness = map(amplitude, AMPLITUDE_THRESHOLD, 2000, 20, 255);
        brightness = constrain(brightness, 0, 255);
    }
    ledcWrite(LED_PWM_CHANNEL, brightness);
    
    // Set flags
    uint16_t flags = 0;
    if (firstPacket) {
        flags |= FLAG_FIRST_PACKET;
        firstPacket = false;
    }
    
    // Send audio packet
    sendAudioPacket(sBuffer, samples_read, flags);
    
    // Periodic status output
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 2000) {
        Serial.print("Amp: ");
        Serial.print(amplitude);
        Serial.print(" | Packets: ");
        Serial.print(packetId);
        Serial.print(" | RSSI: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
        lastStatus = millis();
    }
}

// =============================================================================
// WIFI CONNECTION
// =============================================================================
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
        Serial.println("Check WiFi credentials and restart");
        while (1) {
            ledcWrite(LED_PWM_CHANNEL, 255);
            delay(200);
            ledcWrite(LED_PWM_CHANNEL, 0);
            delay(200);
        }
    }
}

// =============================================================================
// SEND AUDIO PACKET
// =============================================================================
bool sendAudioPacket(int16_t* samples, int count, uint16_t flags) {
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
