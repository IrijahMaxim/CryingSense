# ESP32 Blink Script - Blinks the onboard LED 5 times
# Try to import MicroPython's machine.Pin, else mock for local testing
try:
    from machine import Pin
except ImportError:
    class Pin:
        OUT = 0
        def __init__(self, pin, mode):
            print(f"[MOCK] Pin {pin} initialized in mode {mode}")
        def value(self, v):
            print(f"[MOCK] Pin set to {v}")
import time

# Initialize GPIO 2 (built-in LED on most ESP32 boards)
led = Pin(2, Pin.OUT)

# Blink the LED 5 times
for i in range(5):
    led.value(1)  # Turn LED on
    time.sleep(0.5)
    led.value(0)  # Turn LED off
    time.sleep(0.5)

print("Blink complete!")