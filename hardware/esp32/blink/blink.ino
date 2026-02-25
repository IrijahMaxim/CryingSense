const int ledPin = 2;

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  Serial.println("LED ON");
  digitalWrite(ledPin, HIGH);
  delay(1000);

  Serial.println("LED OFF");
  digitalWrite(ledPin, LOW);
  delay(1000);
}