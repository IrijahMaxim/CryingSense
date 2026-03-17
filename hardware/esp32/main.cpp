#define ENABLE_BLE_CONFIG 0

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Preferences.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#if ENABLE_BLE_CONFIG
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#endif

/*
 * ESP32 INMP441 Audio Transmitter over WiFi UDP
 * 
 * Captures audio from INMP441 microphone and sends to PC/Raspberry Pi
 * via WiFi UDP for real-time cry classification.
 * 
 * HARDWARE CONNECTIONS:
 *   INMP441     ESP32      MB102      <- DC
 *   VDD    ->   3.3V    ->  3.3V
 *   GND    ->   GND     ->  GND
 *   SD     ->   D16
 *   WS     ->   D17
 *   SCK    ->   D18
 *   L/R    ->   GND
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
// WIFI CONFIGURATION (defaults — overridden by NVS-stored values)
// =============================================================================
#define DEFAULT_WIFI_SSID     "PLDTHOMEFIBRE538D"
#define DEFAULT_WIFI_PASSWORD "PLDTWIFI88IEC"
#define DEFAULT_SERVER_IP     "192.168.1.18"
#define DEFAULT_SERVER_PORT   8888

// Mutable credentials (populated from NVS on boot, fallback to defaults)
String wifiSSID;
String wifiPassword;
String serverIPStr;
int    serverPortNum;

#define SERIAL_BAUD_RATE 115200

// =============================================================================
// HTTP CONFIG SERVER
// =============================================================================
WebServer httpServer(80);
volatile bool pendingWiFiReconnect = false;

// =============================================================================
// NVS PERSISTENT STORAGE
// =============================================================================
Preferences preferences;
#define NVS_NAMESPACE "cryingsense"

// =============================================================================
// BLE CONFIGURATION (optional, disabled by default to save flash)
// =============================================================================
#if ENABLE_BLE_CONFIG
#define BLE_DEVICE_NAME       "CryingSense-ESP32"
#define BLE_SERVICE_UUID      "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define BLE_CONFIG_CHAR_UUID  "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define BLE_STATUS_CHAR_UUID  "d1a7e5b2-42c3-4f8e-9d1a-3b5c7e9f0a2d"

BLEServer*         pBLEServer   = nullptr;
BLECharacteristic* pConfigChar  = nullptr;
BLECharacteristic* pStatusChar  = nullptr;
bool bleClientConnected = false;
#endif

// =============================================================================
// TRANSPORT MODES
// =============================================================================
// WiFi UDP remains primary. Serial can be used as fallback or mirror.
#define ENABLE_SERIAL_DEBUG_LOGS 1

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
#define BLUE_LED 4   // WiFi connected indicator
#define RED_LED 5    // WiFi disconnected / no-link indicator

// =============================================================================
// AUDIO PROCESSING
// =============================================================================
#define SOFTWARE_GAIN 1.5              // Amplification factor (increase if needed)

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
#define FLAG_CRY_DETECTED 0x04

// =============================================================================
// GLOBALS
// =============================================================================
WiFiUDP udp;
uint32_t packetId = 0;
bool firstPacket = true;
unsigned long lastWiFiCheck = 0;
const unsigned long WIFI_RETRY_INTERVAL = 10000;
uint32_t udpSendFailures = 0;

int16_t sBuffer[BUFFER_SIZE];
uint8_t txBuffer[HEADER_SIZE + BUFFER_SIZE * 2];  // Header + 16-bit samples

// Forward declarations
void i2s_install();
void i2s_setpin();
void connectWiFi();
bool sendAudioPacket(int16_t* samples, int count, uint16_t flags);
void bootLedSelfTest();
void loadCredentials();
void saveCredentials();
void setupHTTPServer();
void addCorsHeaders();
void handleOptions();
void handleConfigPost();
void handleConfigGet();
void handleGetStatus();
#if ENABLE_BLE_CONFIG
void setupBLE();
#endif
void applyNewConfig(const String& ssid, const String& password,
                    const String& srvIP, int srvPort);

// =============================================================================
// SETUP
// =============================================================================
void setup() {
#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("CryingSense ESP32 Audio Transmitter");
    Serial.println("Mode: WiFi UDP + HTTP Config + BLE");
    Serial.println("========================================");
#endif
    
    // Setup LED
    pinMode(BLUE_LED, OUTPUT);
    pinMode(RED_LED, OUTPUT);

    digitalWrite(BLUE_LED, LOW);
    digitalWrite(RED_LED, LOW);

    bootLedSelfTest();

#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("LED configured");
#endif

    // Load credentials from NVS (falls back to defaults)
    loadCredentials();

    // Start BLE provisioning server (optional)
#if ENABLE_BLE_CONFIG
    setupBLE();
#endif
    
    connectWiFi();

    // Start HTTP config server (responds when WiFi is up)
    setupHTTPServer();
    
    // Initialize I2S
#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("Initializing I2S...");
#endif
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
#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("I2S initialized and buffer cleared");
#endif
    
    udp.begin(8889);
#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("UDP initialized");
    
    Serial.println("\nReady to transmit audio via WiFi UDP!");
    Serial.print("Baud Rate: ");
    Serial.println(SERIAL_BAUD_RATE);
    Serial.print("Software Gain: ");
    Serial.print(SOFTWARE_GAIN);
    Serial.println("x");
    Serial.println("Thresholds - Ambient: 30 | Crying: 100 | Loud: 250");
    Serial.println("========================================\n");
#endif
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
    // Handle pending WiFi reconnect (after config update via HTTP)
    if (pendingWiFiReconnect) {
        pendingWiFiReconnect = false;
        delay(500); // Let HTTP response flush
        connectWiFi();
    }

    // Service HTTP config requests
    httpServer.handleClient();

    unsigned long currentTime = millis();
    if (WiFi.status() != WL_CONNECTED && (currentTime - lastWiFiCheck > WIFI_RETRY_INTERVAL)) {
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println("WiFi disconnected! Attempting reconnect...");
#endif
        connectWiFi();
        lastWiFiCheck = currentTime;
    }
    
    // Read audio from I2S with a short timeout so status LED keeps updating
    // even if the mic is disconnected or miswired.
    size_t bytesIn = 0;
    int32_t raw32Buffer[BUFFER_SIZE];
    int samples_read = 0;
    bool hasAudio = false;

    esp_err_t result = i2s_read(I2S_PORT, &raw32Buffer, BYTES_TO_READ, &bytesIn, 20 / portTICK_PERIOD_MS);

    if (result == ESP_OK && bytesIn > 0) {
        hasAudio = true;
        samples_read = bytesIn / 4;

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
    }

    // Calculate amplitude with spike rejection filter
    long sum_abs = 0;
    int16_t peak = 0;
    int valid_samples = 0;
    
    // First pass: calculate rough average to detect spikes
    if (hasAudio && samples_read > 0) {
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
    }
    
    int amplitude = valid_samples > 0 ? sum_abs / valid_samples : 0;
    
    // ============================
    // LED STATUS SYSTEM
    // ============================

    // Blue LED: solid ON when WiFi is connected
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(BLUE_LED, HIGH);
    } else {
        digitalWrite(BLUE_LED, LOW);
    }

    // Red LED: solid ON when WiFi is NOT connected (no-link / unconnected state)
    if (WiFi.status() != WL_CONNECTED) {
        digitalWrite(RED_LED, HIGH);
    } else {
        digitalWrite(RED_LED, LOW);
    }
    
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
    
    bool sentUdp = false;
    if (WiFi.status() == WL_CONNECTED) {
        sentUdp = sendAudioPacket(sBuffer, samples_read, flags);
        if (!sentUdp) {
            udpSendFailures++;
        }
    }
    
    // Periodic status output
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 2000) {
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.print("Amp: ");
        Serial.print(amplitude);
        Serial.print(" | Peak: ");
        Serial.print(peak);
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
        Serial.print(" | TX: ");
        if (sentUdp) {
            Serial.print("UDP");
        } else {
            Serial.print("NO_LINK");
        }
        Serial.print(" | WiFi: ");
        if (WiFi.status() == WL_CONNECTED) {
            Serial.print("OK");
            Serial.print(" | RSSI: ");
            Serial.print(WiFi.RSSI());
            Serial.print(" dBm");
        } else {
            Serial.print("OFFLINE");
        }
        Serial.print(" | UDP_FAIL: ");
        Serial.print(udpSendFailures);
        Serial.println();
#endif
        lastStatus = millis();
    }
}

// =============================================================================
// WIFI CONNECTION
// =============================================================================
void connectWiFi() {
#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.print("Connecting to WiFi: ");
    Serial.println(wifiSSID);
#endif
    
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(wifiSSID.c_str(), wifiPassword.c_str());
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
    #if ENABLE_SERIAL_DEBUG_LOGS
        Serial.print(".");
    #endif
        attempts++;
        
        // Blink LED while connecting
        digitalWrite(BLUE_LED, attempts % 2);
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        // Re-bind UDP after (re)connect so the sender socket is always valid.
        udp.stop();
        udp.begin(8889);
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println(" Connected!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        Serial.print("MAC Address: ");
        Serial.println(WiFi.macAddress());
#endif
        digitalWrite(BLUE_LED, LOW);
    } else {
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println(" Failed!");
        Serial.println("Continuing in offline mode");
        Serial.println("Will retry WiFi connection in loop...");
#endif
        digitalWrite(BLUE_LED, LOW);
    }
}

void bootLedSelfTest() {
    for (int i = 0; i < 3; i++) {
        digitalWrite(BLUE_LED, HIGH);
        digitalWrite(RED_LED, HIGH);
        delay(120);
        digitalWrite(BLUE_LED, LOW);
        digitalWrite(RED_LED, LOW);
        delay(120);
    }
}

// =============================================================================
// SEND AUDIO PACKET VIA UDP
// =============================================================================
bool sendAudioPacket(int16_t* samples, int count, uint16_t flags) {
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }

    uint32_t timestamp = millis();

    memcpy(txBuffer, &packetId, 4);
    memcpy(txBuffer + 4, &timestamp, 4);
    memcpy(txBuffer + 8, &count, 2);
    memcpy(txBuffer + 10, &flags, 2);
    memcpy(txBuffer + HEADER_SIZE, samples, count * 2);

    int totalSize = HEADER_SIZE + (count * 2);
    udp.beginPacket(serverIPStr.c_str(), serverPortNum);
    size_t written = udp.write(txBuffer, totalSize);
    int result = udp.endPacket();

    packetId++;

    return (result == 1 && written == (size_t)totalSize);
}

// =============================================================================
// SEND AUDIO PACKET VIA SERIAL (COM PORT)
// =============================================================================
/* Serial audio fallback removed to reduce unused code. */

// =============================================================================
// I2S CONFIGURATION
// =============================================================================
void i2s_install() {
    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = i2s_bits_per_sample_t(32),
        // INMP441 L/R=GND → left-channel slot (WS LOW).
        // ESP32's I2S driver names are inverted vs. standard I2S: ONLY_RIGHT
        // captures the WS-LOW (left) slot, which is what we need here.
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
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

// =============================================================================
// NVS CREDENTIAL STORAGE
// =============================================================================
void loadCredentials() {
    preferences.begin(NVS_NAMESPACE, true); // read-only
    wifiSSID      = preferences.getString("ssid",      DEFAULT_WIFI_SSID);
    wifiPassword   = preferences.getString("password",  DEFAULT_WIFI_PASSWORD);
    serverIPStr   = preferences.getString("server_ip", DEFAULT_SERVER_IP);
    serverPortNum = preferences.getInt("server_port",  DEFAULT_SERVER_PORT);
    preferences.end();

#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("Credentials loaded from NVS:");
    Serial.print("  SSID: ");       Serial.println(wifiSSID);
    Serial.print("  Server IP: ");  Serial.println(serverIPStr);
    Serial.print("  Server Port: ");Serial.println(serverPortNum);
#endif
}

void saveCredentials() {
    preferences.begin(NVS_NAMESPACE, false); // read-write
    preferences.putString("ssid",        wifiSSID);
    preferences.putString("password",    wifiPassword);
    preferences.putString("server_ip",   serverIPStr);
    preferences.putInt("server_port",    serverPortNum);
    preferences.end();

#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("Credentials saved to NVS");
#endif
}

// =============================================================================
// APPLY NEW CONFIG (shared by HTTP and BLE paths)
// =============================================================================
void applyNewConfig(const String& ssid, const String& password,
                    const String& srvIP, int srvPort) {
    bool needsReconnect = (ssid != wifiSSID || password != wifiPassword);

    wifiSSID      = ssid;
    wifiPassword   = password;
    serverIPStr   = srvIP;
    serverPortNum = srvPort;

    saveCredentials();

    if (needsReconnect) {
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println("WiFi credentials changed — scheduling reconnect");
#endif
        pendingWiFiReconnect = true;
    } else {
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println("Server target updated (no WiFi reconnect needed)");
#endif
    }
}

// =============================================================================
// HTTP CONFIG SERVER
// =============================================================================
void addCorsHeaders() {
    httpServer.sendHeader("Access-Control-Allow-Origin", "*");
    httpServer.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    httpServer.sendHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    // Required by Chromium Private Network Access preflight when web app
    // (for example localhost) calls a private IP like 192.168.x.x.
    httpServer.sendHeader("Access-Control-Allow-Private-Network", "true");
}

void handleOptions() {
    addCorsHeaders();
    httpServer.send(204);
}

void handleConfigPost() {
    addCorsHeaders();
    String newSSID;
    String newPassword;
    String newServerIP;
    int    newPort = DEFAULT_SERVER_PORT;

    if (httpServer.hasArg("plain")) {
        JsonDocument doc;
        DeserializationError err = deserializeJson(doc, httpServer.arg("plain"));
        if (err) {
            httpServer.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
            return;
        }

        newSSID     = doc["ssid"]      | "";
        newPassword = doc["password"]  | "";
        newServerIP = doc["server_ip"] | "";
        newPort     = doc["server_port"] | DEFAULT_SERVER_PORT;
    } else {
        // Fallback for simple form posts (no JSON body)
        newSSID     = httpServer.arg("ssid");
        newPassword = httpServer.arg("password");
        newServerIP = httpServer.arg("server_ip");
        if (httpServer.hasArg("server_port")) {
            newPort = httpServer.arg("server_port").toInt();
        }
    }

    if (newSSID.length() == 0) {
        httpServer.send(400, "application/json", "{\"status\":\"error\",\"message\":\"SSID required\"}");
        return;
    }

    // Build response BEFORE applying (since reconnect may drop connection)
    JsonDocument resp;
    bool willReconnect = (newSSID != wifiSSID || newPassword != wifiPassword);
    resp["status"]  = "ok";
    resp["ip"]      = WiFi.localIP().toString();
    resp["message"] = willReconnect ? "Config saved. Reconnecting WiFi..." : "Config saved.";
    String respStr;
    serializeJson(resp, respStr);
    httpServer.send(200, "application/json", respStr);

    // Now apply (may trigger reconnect on next loop iteration)
    applyNewConfig(newSSID, newPassword, newServerIP, newPort);
}

void handleConfigGet() {
    addCorsHeaders();

    JsonDocument doc;
    doc["ssid"]        = wifiSSID;
    doc["server_ip"]   = serverIPStr;
    doc["server_port"] = serverPortNum;
    doc["wifi"]        = (WiFi.status() == WL_CONNECTED) ? "connected" : "disconnected";
    doc["ip"]          = WiFi.localIP().toString();
    String out;
    serializeJson(doc, out);
    httpServer.send(200, "application/json", out);
}

void handleGetStatus() {
    addCorsHeaders();
    JsonDocument doc;
    doc["wifi"]        = (WiFi.status() == WL_CONNECTED) ? "connected" : "disconnected";
    doc["ip"]          = WiFi.localIP().toString();
    doc["rssi"]        = WiFi.RSSI();
    doc["ssid"]        = wifiSSID;
    doc["server_ip"]   = serverIPStr;
    doc["server_port"] = serverPortNum;
    doc["packets"]     = packetId;
    doc["uptime_ms"]   = millis();
    String out;
    serializeJson(doc, out);
    httpServer.send(200, "application/json", out);
}

void setupHTTPServer() {
    httpServer.on("/config", HTTP_POST, handleConfigPost);
    httpServer.on("/config", HTTP_GET, handleConfigGet);
    httpServer.on("/config", HTTP_OPTIONS, handleOptions);
    httpServer.on("/status", HTTP_GET,  handleGetStatus);
    httpServer.on("/status", HTTP_OPTIONS, handleOptions);
    httpServer.begin();

#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.println("HTTP config server started on port 80");
#endif
}

// =============================================================================
// BLE CONFIGURATION RECEIVER
// =============================================================================
#if ENABLE_BLE_CONFIG
class CryingSenseBLEServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* server) override {
        bleClientConnected = true;
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println("BLE client connected");
#endif
    }

    void onDisconnect(BLEServer* server) override {
        bleClientConnected = false;
#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.println("BLE client disconnected");
#endif
    // Restart advertising so another client can connect.
    BLEDevice::startAdvertising();
    }
};

class ConfigCharCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic) override {
        std::string raw = pCharacteristic->getValue();
        if (raw.length() == 0) return;

        String value = String(raw.c_str());

#if ENABLE_SERIAL_DEBUG_LOGS
        Serial.print("BLE config received (");
        Serial.print(value.length());
        Serial.println(" bytes)");
#endif

        JsonDocument doc;
        DeserializationError err = deserializeJson(doc, value);
        if (err) {
#if ENABLE_SERIAL_DEBUG_LOGS
            Serial.println("BLE JSON parse error");
#endif
            if (pStatusChar) {
                pStatusChar->setValue("{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
                pStatusChar->notify();
            }
            return;
        }

        String newSSID     = doc["ssid"]      | "";
        String newPassword = doc["password"]  | "";
        String newServerIP = doc["server_ip"] | "";
        int    newPort     = doc["server_port"] | DEFAULT_SERVER_PORT;

        if (newSSID.length() == 0) {
            if (pStatusChar) {
                pStatusChar->setValue("{\"status\":\"error\",\"message\":\"SSID required\"}");
                pStatusChar->notify();
            }
            return;
        }

        applyNewConfig(newSSID, newPassword, newServerIP, newPort);

        // Notify BLE client of success
        if (pStatusChar) {
            JsonDocument resp;
            resp["status"]  = "ok";
            resp["message"] = "Config saved. Reconnecting WiFi...";
            String respStr;
            serializeJson(resp, respStr);
            pStatusChar->setValue(respStr.c_str());
            pStatusChar->notify();
        }
    }
};

void setupBLE() {
    BLEDevice::init(BLE_DEVICE_NAME);
    pBLEServer = BLEDevice::createServer();
    pBLEServer->setCallbacks(new CryingSenseBLEServerCallbacks());

    BLEService* pService = pBLEServer->createService(BLE_SERVICE_UUID);

    // Config characteristic — Android writes JSON here
    pConfigChar = pService->createCharacteristic(
        BLE_CONFIG_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE
    );
    pConfigChar->setCallbacks(new ConfigCharCallbacks());

    // Status characteristic — ESP32 notifies Android of results
    pStatusChar = pService->createCharacteristic(
        BLE_STATUS_CHAR_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pStatusChar->addDescriptor(new BLE2902());

    pService->start();

    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->start();

#if ENABLE_SERIAL_DEBUG_LOGS
    Serial.print("BLE advertising as: ");
    Serial.println(BLE_DEVICE_NAME);
#endif
}
#endif
