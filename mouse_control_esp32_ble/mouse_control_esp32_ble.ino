
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================
// HARDWARE CONFIGURATION
// ============================================================

#define MPU6050_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22
#define BUTTON_PIN 4

// ============================================================
// BLUETOOTH CONFIGURATION
// Must match the Python controller
// ============================================================

#define DEVICE_NAME "GestureAssist-BLE"

#define SERVICE_UUID \
  "7f510001-1b15-4d9c-8f8a-7e9d4c2a1000"

#define CHARACTERISTIC_UUID \
  "7f510002-1b15-4d9c-8f8a-7e9d4c2a1000"

// ============================================================
// GLOBAL VARIABLES
// ============================================================

BLEServer *bleServer = nullptr;
BLECharacteristic *sensorCharacteristic = nullptr;

volatile bool deviceConnected = false;

// ============================================================
// BLE SERVER CALLBACKS
// ============================================================

class GestureServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    deviceConnected = true;
    Serial.println("BLE client connected");
  }

  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    Serial.println("BLE client disconnected");

    delay(200);
    BLEDevice::startAdvertising();

    Serial.println("BLE advertising restarted");
  }
};

// ============================================================
// MPU6050 INITIALIZATION
// ============================================================

bool initializeMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);  // Power management register
  Wire.write(0x00);  // Wake up MPU6050

  return Wire.endTransmission() == 0;
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("GestureAssist ESP32 Starting");
  Serial.println("================================");

  // Initialize I2C and button
  Wire.begin(SDA_PIN, SCL_PIN);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Initialize MPU6050
  if (!initializeMPU6050()) {
    Serial.println("ERROR: MPU6050 not detected!");
    Serial.println("Check SDA, SCL, VCC and GND.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 initialized");

  // Initialize BLE
  BLEDevice::init(DEVICE_NAME);

  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new GestureServerCallbacks());

  BLEService *service =
      bleServer->createService(SERVICE_UUID);

  // Create the sensor characteristic
  sensorCharacteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ |
      BLECharacteristic::PROPERTY_NOTIFY
  );

  // Required for BLE notification subscription
  BLE2902 *descriptor = new BLE2902();
  descriptor->setNotifications(true);
  sensorCharacteristic->addDescriptor(descriptor);

  sensorCharacteristic->setValue("0,0,0");

  // Start BLE service
  service->start();

  // Configure advertising
  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMinPreferred(0x12);

  BLEDevice::startAdvertising();

  Serial.println("BLE advertising started");
  Serial.print("Device name: ");
  Serial.println(DEVICE_NAME);

  Serial.println("Service UUID: " SERVICE_UUID);
  Serial.println("Characteristic UUID: " CHARACTERISTIC_UUID);
  Serial.println("Data format: X,Y,BUTTON");
  Serial.println("Waiting for Python connection...");
}

// ============================================================
// READ MPU6050 ACCELEROMETER
// ============================================================

bool readSensor(int16_t &x, int16_t &y) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);  // Accelerometer X high byte

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU6050_ADDR, 6, true) != 6) {
    return false;
  }

  x = (int16_t)((Wire.read() << 8) | Wire.read());
  y = (int16_t)((Wire.read() << 8) | Wire.read());

  // Discard Z-axis bytes
  Wire.read();
  Wire.read();

  return true;
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  int16_t accelX;
  int16_t accelY;

  if (!readSensor(accelX, accelY)) {
    delay(20);
    return;
  }

  // Button connected to GND when pressed
  int button = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;

  // Create CSV message: X,Y,BUTTON
  char data[32];

  snprintf(
      data,
      sizeof(data),
      "%d,%d,%d",
      (int)accelX,
      (int)accelY,
      button
  );

  // USB Serial Monitor debugging
  Serial.println(data);

  // Send sensor readings through BLE notifications
  if (deviceConnected && sensorCharacteristic != nullptr) {
    sensorCharacteristic->setValue(
        (uint8_t *)data,
        strlen(data)
    );

    sensorCharacteristic->notify();
  }

  delay(20);
}
