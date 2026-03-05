#include <Arduino.h>
#include <driver/i2s.h>

/*
 * ESP32 INMP441 Microphone Audio Capture
 * 
 * HARDWARE TIPS TO REDUCE NOISE/SPIKES:
 * 1. Use a stable 3.3V power supply (not USB 5V with voltage regulator if possible)
 * 2. Add 10uF capacitor between VDD and GND on INMP441 (close to mic)
 * 3. Add 100nF ceramic capacitor between VDD and GND on ESP32
 * 4. Keep I2S wires short and away from power lines
 * 5. Use common ground between ESP32 and INMP441
 * 6. Twist I2S data wires together to reduce interference
 */

// Forward declarations
void i2s_install();
void i2s_setpin();

// I2S Microphone Pin Configuration
#define I2S_WS    17  // D17 - Word Select (LRCLK)
#define I2S_SD    16  // D16 - Serial Data (DOUT)
#define I2S_SCK   18  // D18 - Serial Clock (BCLK)

// LED Configuration
#define LED_PIN   2   // Built-in LED (change if using external LED)
#define LED_PWM_CHANNEL 0
#define LED_PWM_FREQ 5000
#define LED_PWM_RESOLUTION 8  // 8-bit resolution (0-255)

// SENSITIVITY ADJUSTMENT - Increase this to make microphone more sensitive
#define SOFTWARE_GAIN 1.0          // Amplification factor (1.0 = no gain, increase if needed)

// Baby Crying Detection thresholds (adjusted for proper bit extraction)
#define AMBIENT_THRESHOLD 30.0     // Normal room noise level
#define CRYING_THRESHOLD 100.0     // Baby crying detection threshold  
#define LOUD_CRY_THRESHOLD 250.0   // Very loud crying
#define MAX_AMPLITUDE 600.0        // Maximum expected amplitude

// Sustained sound detection (to filter brief noises)
#define DETECTION_COUNT 2          // Reduced to 2 for faster response (was 3)
int loudCount = 0;                 // Counter for sustained detection

// Visualization mode - output raw samples for better waveform display
#define OUTPUT_RAW_SAMPLES true    // Set to false for minimal output
#define SAMPLES_TO_OUTPUT 32       // Number of samples to output per buffer

#define I2S_PORT I2S_NUM_0
#define BUFFER_SIZE 512  // Number of samples
#define BYTES_TO_READ (BUFFER_SIZE * 4)  // 4 bytes per 32-bit sample

int16_t sBuffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  Serial.println("Setup I2S ...");

  // Setup LED PWM
  ledcSetup(LED_PWM_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
  ledcAttachPin(LED_PIN, LED_PWM_CHANNEL);
  ledcWrite(LED_PWM_CHANNEL, 0);  // Start with LED off
  Serial.println("LED configured");

  delay(1000);
  i2s_install();
  i2s_setpin();
  i2s_start(I2S_PORT);
  delay(500);
  
  // Clear initial I2S buffer to remove startup zeros and stabilize
  size_t bytes_read;
  int32_t dummy_buffer[BUFFER_SIZE];
  for (int i = 0; i < 5; i++) {
    i2s_read(I2S_PORT, &dummy_buffer, BYTES_TO_READ, &bytes_read, 100);
    delay(10);
  }
  Serial.println("I2S buffer cleared and stabilized");
  
  Serial.println("Baby Cry Monitor Ready!");
  Serial.println("LED brightness indicates cry intensity");
  Serial.print("Software Gain: ");
  Serial.print(SOFTWARE_GAIN);
  Serial.println("x");
  Serial.println("Thresholds - Ambient: 30 | Crying: 100 | Loud: 250");
}

void loop() {
  size_t bytesIn = 0;
  int32_t raw32Buffer[BUFFER_SIZE];  // Temporary buffer for 32-bit samples
  
  esp_err_t result = i2s_read(I2S_PORT, &raw32Buffer, BYTES_TO_READ, &bytesIn, portMAX_DELAY);
  
  if (result == ESP_OK && bytesIn > 0)
  {
    // Correct calculation: 32-bit samples = 4 bytes each
    int samples_read = bytesIn / 4;
    
    if (samples_read > 0) {
      // Convert 32-bit samples to 16-bit for processing
      // INMP441 outputs 18-bit data left-aligned in 32-bit word
      for (int i = 0; i < samples_read; ++i) {
        // Shift right by 16 bits to get the most significant 16 bits
        sBuffer[i] = (int16_t)(raw32Buffer[i] >> 16);
      }
      
      // Remove DC offset (helps with power supply noise)
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
      
      // Calculate amplitude using simple average absolute value
      // Also apply spike rejection filter
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
      
      // Map amplitude to LED brightness (0-255)
      int brightness = 0;
      if (amplitude > AMBIENT_THRESHOLD) {
        brightness = (int)((amplitude / MAX_AMPLITUDE) * 255.0);
        brightness = constrain(brightness, 0, 255);
      }
      
      // Update LED brightness
      ledcWrite(LED_PWM_CHANNEL, brightness);
      
      // Baby cry detection with sustained sound check
      if (amplitude > CRYING_THRESHOLD) {
        loudCount++;
        
        // Only trigger if sustained for multiple readings (reduces false positives)
        if (loudCount >= DETECTION_COUNT) {
          if (amplitude > LOUD_CRY_THRESHOLD) {
            Serial.println("🚨 BABY CRYING LOUDLY! 🚨");
          } else {
            Serial.println("⚠️  Baby crying detected");
          }
          loudCount = DETECTION_COUNT; // Cap the counter
        }
      } else {
        // Reset counter if sound drops below threshold
        if (loudCount > 0) {
          loudCount--;
        }
      }
      
      // Print current level for monitoring
      Serial.print("Amp: ");
      Serial.print(amplitude);
      Serial.print(" | Peak: ");
      Serial.print(peak);
      Serial.print(" | LED: ");
      Serial.print(brightness);
      Serial.print(" | Status: ");
      if (amplitude < AMBIENT_THRESHOLD) {
        Serial.println("Quiet");
      } else if (amplitude < CRYING_THRESHOLD) {
        Serial.println("Normal");
      } else if (amplitude < LOUD_CRY_THRESHOLD) {
        Serial.println("CRYING");
      } else {
        Serial.println("LOUD CRY!");
      }
      
      // Output raw samples for visualization
      #if OUTPUT_RAW_SAMPLES
      Serial.print("SAMPLES:");
      int step = samples_read / SAMPLES_TO_OUTPUT;
      if (step < 1) step = 1;
      
      // Apply simple 3-point median filter to output samples for cleaner visualization
      for (int i = 0; i < samples_read && i < SAMPLES_TO_OUTPUT * step; i += step) {
        int16_t sample;
        if (i == 0 || i >= samples_read - step) {
          sample = sBuffer[i];  // Edge case: use original
        } else {
          // Median of 3 consecutive samples
          int16_t a = sBuffer[i - step];
          int16_t b = sBuffer[i];
          int16_t c = sBuffer[i + step];
          // Simple median without sorting
          sample = max(min(a, b), min(max(a, b), c));
        }
        Serial.print(sample);
        if (i + step < samples_read) Serial.print(",");
      }
      Serial.println();
      #endif
    }
  }
}

void i2s_install(){
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = i2s_bits_per_sample_t(32),  // 32-bit for better dynamic range
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),  // Standard I2S format for INMP441
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,  // Higher priority interrupt
    .dma_buf_count = 4,                        // Reduced for lower latency
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin(){
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_set_pin(I2S_PORT, &pin_config);
}