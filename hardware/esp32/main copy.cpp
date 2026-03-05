/*#include <Arduino.h>
#include <driver/i2s.h>

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

// Volume thresholds
#define LOUD_THRESHOLD 2000.0  // Adjust based on your microphone sensitivity
#define MAX_AMPLITUDE 8000.0   // Maximum expected amplitude for scaling

#define I2S_PORT I2S_NUM_0
#define BUFFER_SIZE 512  // Number of samples
#define BYTES_TO_READ (BUFFER_SIZE * 2)  // 2 bytes per 16-bit sample

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
  
  Serial.println("System ready - LED will light up with noise!");
}

void loop() {
  size_t bytesIn = 0;
  esp_err_t result = i2s_read(I2S_PORT, &sBuffer, BYTES_TO_READ, &bytesIn, portMAX_DELAY);
  
  if (result == ESP_OK && bytesIn > 0)
  {
    // Correct calculation: 16-bit samples = 2 bytes each
    int samples_read = bytesIn / 2;
    
    if (samples_read > 0) {
      // Calculate amplitude (RMS for better volume representation)
      float sum_squares = 0;
      for (int i = 0; i < samples_read; ++i) {
        sum_squares += (float)sBuffer[i] * sBuffer[i];
      }
      float rms = sqrt(sum_squares / samples_read);
      
      // Map amplitude to LED brightness (0-255)
      int brightness = (int)((rms / MAX_AMPLITUDE) * 255.0);
      brightness = constrain(brightness, 0, 255);  // Ensure within valid range
      
      // Update LED brightness
      ledcWrite(LED_PWM_CHANNEL, brightness);
      
      // Check if too loud and send notification
      if (rms > LOUD_THRESHOLD) {
        Serial.println("TOO LOUD!!");
      }
      
      // Print current level for monitoring
      Serial.print("Volume: ");
      Serial.print(rms);
      Serial.print(" | LED: ");
      Serial.println(brightness);
    }
  }
}

void i2s_install(){
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0, // default interrupt priority
    .dma_buf_count = 8,
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
}*/