#include <Arduino.h>
// #include <WiFi.h>        // COMMENTED OUT - Using COM port instead
// #include <WiFiUdp.h>     // COMMENTED OUT - Using COM port instead
#include <driver/i2s.h>

/*
 * ESP32 INMP441 Audio Transmitter over COM Port (Serial)
 *
 * Captures audio from INMP441 microphone and sends to PC/Raspberry Pi
 * via Serial (USB) for real-time cry classification.
 */

// =============================================================================
// WIFI CONFIGURATION - COMMENTED OUT (Using COM port)
// =============================================================================
// const char* WIFI_SSID = "YOUR_WIFI_SSID";
// const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// const char* SERVER_IP = "192.168.1.9";
// const int SERVER_PORT = 8888;

// =============================================================================
// SERIAL CONFIGURATION
// =============================================================================
#define SERIAL_BAUD_RATE 115200
#define SERIAL_SYNC_BYTE_1 0xAA
#define SERIAL_SYNC_BYTE_2 0x55

// =============================================================================
// I2S MICROPHONE CONFIGURATION
// =============================================================================
#define I2S_WS    17
#define I2S_SD    16
#define I2S_SCK   18
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
#define AMBIENT_THRESHOLD 30.0
#define CRYING_THRESHOLD 100.0
#define LOUD_CRY_THRESHOLD 250.0
#define MAX_AMPLITUDE 600.0
#define DETECTION_COUNT 2
int loudCount = 0;

// =============================================================================
// PACKET PROTOCOL
// =============================================================================
#define HEADER_SIZE 12
#define FLAG_FIRST_PACKET 0x01
#define FLAG_LAST_PACKET 0x02
#define FLAG_CRY_DETECTED 0x04

// =============================================================================
// GLOBALS
// =============================================================================
uint32_t packetId = 0;
bool firstPacket = true;

int16_t sBuffer[BUFFER_SIZE];

// Forward declarations
void i2s_install();
void i2s_setpin();
bool sendAudioPacketSerial(int16_t* samples, int count, uint16_t flags);

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("CryingSense ESP32 Audio Transmitter");
    Serial.println("Mode: COM Port (Serial)");
    Serial.println("========================================");

    ledcSetup(LED_PWM_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
    ledcAttachPin(LED_PIN, LED_PWM_CHANNEL);
    ledcWrite(LED_PWM_CHANNEL, 0);

    i2s_install();
    i2s_setpin();
    i2s_start(I2S_PORT);

    size_t bytes_read;
    int32_t dummy[BUFFER_SIZE];
    for (int i = 0; i < 5; i++) {
        i2s_read(I2S_PORT, &dummy, BYTES_TO_READ, &bytes_read, 100);
        delay(10);
    }

    Serial.println("Ready to transmit audio via Serial");
}

void loop() {
    size_t bytesIn = 0;
    int32_t raw32Buffer[BUFFER_SIZE];

    esp_err_t result = i2s_read(I2S_PORT, &raw32Buffer, BYTES_TO_READ, &bytesIn, portMAX_DELAY);
    if (result != ESP_OK || bytesIn == 0) {
        return;
    }

    int samples_read = bytesIn / 4;
    for (int i = 0; i < samples_read; ++i) {
        sBuffer[i] = (int16_t)(raw32Buffer[i] >> 16);
    }

    long dc_sum = 0;
    for (int i = 0; i < samples_read; ++i) {
        dc_sum += sBuffer[i];
    }
    int16_t dc_offset = dc_sum / samples_read;

    for (int i = 0; i < samples_read; ++i) {
        sBuffer[i] = sBuffer[i] - dc_offset;
        float amplified = (float)sBuffer[i] * SOFTWARE_GAIN;
        amplified = constrain(amplified, -32768, 32767);
        sBuffer[i] = (int16_t)amplified;
    }

    long rough_sum = 0;
    for (int i = 0; i < samples_read; ++i) {
        rough_sum += abs(sBuffer[i]);
    }
    int rough_avg = rough_sum / samples_read;
    int spike_threshold = rough_avg * 10;

    long sum_abs = 0;
    int16_t peak = 0;
    int valid_samples = 0;
    for (int i = 0; i < samples_read; ++i) {
        int16_t abs_val = abs(sBuffer[i]);
        if (abs_val < spike_threshold || spike_threshold == 0) {
            sum_abs += abs_val;
            valid_samples++;
            if (abs_val > peak) peak = abs_val;
        }
    }

    int amplitude = valid_samples > 0 ? sum_abs / valid_samples : 0;
    int brightness = 0;
    if (amplitude > AMBIENT_THRESHOLD) {
        brightness = (int)((amplitude / MAX_AMPLITUDE) * 255.0);
        brightness = constrain(brightness, 0, 255);
    }
    ledcWrite(LED_PWM_CHANNEL, brightness);

    if (amplitude > CRYING_THRESHOLD) {
        loudCount++;
        if (loudCount >= DETECTION_COUNT) {
            loudCount = DETECTION_COUNT;
        }
    } else if (loudCount > 0) {
        loudCount--;
    }

    uint16_t flags = 0;
    if (firstPacket) {
        flags |= FLAG_FIRST_PACKET;
        firstPacket = false;
    }
    if (loudCount >= DETECTION_COUNT) {
        flags |= FLAG_CRY_DETECTED;
    }

    sendAudioPacketSerial(sBuffer, samples_read, flags);

    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 2000) {
        Serial.print("Amp: ");
        Serial.print(amplitude);
        Serial.print(" | Peak: ");
        Serial.print(peak);
        Serial.print(" | Packets: ");
        Serial.println(packetId);
        lastStatus = millis();
    }
}

bool sendAudioPacketSerial(int16_t* samples, int count, uint16_t flags) {
    uint32_t timestamp = millis();

    Serial.write(SERIAL_SYNC_BYTE_1);
    Serial.write(SERIAL_SYNC_BYTE_2);
    Serial.write((uint8_t*)&packetId, 4);
    Serial.write((uint8_t*)&timestamp, 4);
    Serial.write((uint8_t*)&count, 2);
    Serial.write((uint8_t*)&flags, 2);
    Serial.write((uint8_t*)samples, count * 2);

    packetId++;
    return true;
}

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
