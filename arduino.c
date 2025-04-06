#include <LiquidCrystal.h>
#include <DHT.h>

#define DHTPIN A1      
#define DHTTYPE DHT11  
#define CO2_THRESHOLD 800    // Updated threshold for CO2
#define TEMP_THRESHOLD 34
#define ALARM_CO2 1000       // CO2 alarm threshold for KPIv
#define VN 2.0               // Predefined value when CO2 exceeds threshold

// CO2 to person estimation parameters
#define BASE_CO2 400         // Baseline outdoor CO2 (ppm)
#define CO2_PER_PERSON 20    // Approximate CO2 contributed per person (ppm)

String uno;
LiquidCrystal lcd(2, 3, 4, 5, 6, 7);
DHT dht(DHTPIN, DHTTYPE);

int in = 8;   
int out = 9;  
int co2 = A0;
int relay = 11;
int buzzer = 10;

int in_count = 0;   
int out_count = 0;  

// Model coefficients (can be updated by Raspberry Pi)
float co2_weight = 0.6;
float temp_weight = 0.3;
float humidity_weight = 0.1;
float trend_threshold = 0.7;

// Buffers for simple moving average
#define BUFFER_SIZE 5
int co2_buffer[BUFFER_SIZE];
int temp_buffer[BUFFER_SIZE];
int buffer_index = 0;
bool buffer_filled = false;

// Variables to track previous sensor states
int prev_ir_in = 0;
int prev_ir_out = 0;

// KPIv calculation variables
float kpiv = 0.0;
int person_co2 = 0;

void setup() {
  dht.begin();
  Serial.begin(9600);
  lcd.begin(16, 2);

  lcd.setCursor(0, 0);
  lcd.print("Edge-Driven CO2");
  lcd.setCursor(0, 1);
  lcd.print("Prediction");
  delay(3000);
  lcd.clear();

  pinMode(in, INPUT);
  pinMode(out, INPUT);
  pinMode(co2, INPUT);
  pinMode(relay, OUTPUT);
  pinMode(buzzer, OUTPUT);

  digitalWrite(relay, HIGH);
  digitalWrite(buzzer, LOW);
  
  // Initialize buffers
  for(int i = 0; i < BUFFER_SIZE; i++) {
    co2_buffer[i] = 0;
    temp_buffer[i] = 0;
  }
}

// Calculate moving average
int calculateMovingAverage(int buffer[], int size) {
  long sum = 0;
  int count = buffer_filled ? BUFFER_SIZE : buffer_index;
  
  for(int i = 0; i < count; i++) {
    sum += buffer[i];
  }
  
  return (count > 0) ? (sum / count) : 0;
}

// Lightweight prediction model
float predictTrend(int current_co2, int current_temp, int current_humidity, int people_count) {
  // Get averages
  int avg_co2 = calculateMovingAverage(co2_buffer, BUFFER_SIZE);
  int avg_temp = calculateMovingAverage(temp_buffer, BUFFER_SIZE);
  
  // Calculate trends (positive = rising, negative = falling)
  float co2_trend = current_co2 - avg_co2;
  float temp_trend = current_temp - avg_temp;
  
  // Normalize trends
  co2_trend = co2_trend / 50.0;  // Normalize by expected variation
  temp_trend = temp_trend / 2.0;  // Normalize by expected variation
  
  // Apply weights
  float weighted_trend = (co2_weight * co2_trend) + 
                         (temp_weight * temp_trend) + 
                         (humidity_weight * (people_count > 0 ? 1 : -1));
  
  return weighted_trend;
}

// Calculate KPIv (Ventilation KPI)
float calculateKPIv(int carbon_value, int actual_people) {
  // If CO2 exceeds alarm threshold, return Vn
  if (carbon_value > ALARM_CO2) {
    return VN;
  }
  
  // Estimate people based on CO2 (simplified model)
  int estimated_people = (carbon_value - BASE_CO2) / CO2_PER_PERSON;
  if (estimated_people < 0) estimated_people = 0;
  
  // Calculate KPIv as ratio of estimated to actual people
  if (actual_people > 0) {
    return (float)estimated_people / actual_people;
  } else {
    return 0; // If no people, ventilation is good
  }
}

// Check for model updates from Raspberry Pi
void checkForModelUpdates() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    
    if (data.startsWith("MODEL:")) {
      // Format: MODEL:co2_weight:0.5,temp_weight:0.3,humidity_weight:0.2,trend_threshold:0.7
      data = data.substring(6); // Remove "MODEL:"
      
      int start = 0;
      int end = data.indexOf(',');
      
      while (end != -1) {
        String param = data.substring(start, end);
        updateParameter(param);
        
        start = end + 1;
        end = data.indexOf(',', start);
      }
      
      // Process the last parameter
      updateParameter(data.substring(start));
      
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Model Updated!");
      delay(1000);
    }
  }
}

// Update model parameter
void updateParameter(String param) {
  int colon = param.indexOf(':');
  String name = param.substring(0, colon);
  float value = param.substring(colon + 1).toFloat();
  
  if (name == "co2_weight") {
    co2_weight = value;
  } else if (name == "temp_weight") {
    temp_weight = value;
  } else if (name == "humidity_weight") {
    humidity_weight = value;
  } else if (name == "trend_threshold") {
    trend_threshold = value;
  }
}

void loop() {
  // Check for model updates
  checkForModelUpdates();
  
  int temperature = dht.readTemperature();
  int humidity = dht.readHumidity();
  int carbon_value = analogRead(co2);
  
  // Update buffers for moving average
  co2_buffer[buffer_index] = carbon_value;
  temp_buffer[buffer_index] = temperature;
  
  buffer_index = (buffer_index + 1) % BUFFER_SIZE;
  if (buffer_index == 0) buffer_filled = true;

  // Read IR Sensors
  int ir_in = digitalRead(in);
  int ir_out = digitalRead(out);

  // Detect Rising Edge for IN sensor
  if (ir_in == 1 && prev_ir_in == 0) {
    in_count++;
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("IN Count: ");
    lcd.print(in_count);
    delay(1000);
  }
  prev_ir_in = ir_in;

  // Detect Rising Edge for OUT sensor
  if (ir_out == 1 && prev_ir_out == 0) {
    out_count++;
    if (in_count > 0) in_count--;  // Ensure IN count doesn't go below zero

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("OUT Count: ");
    lcd.print(out_count);

    lcd.setCursor(0, 1);
    lcd.print("IN Count: ");
    lcd.print(in_count);
    delay(1000);
  }
  prev_ir_out = ir_out;
  
  // Calculate actual people in room
  int people_count = in_count;
  
  // Predict trend using lightweight model
  float trend = predictTrend(carbon_value, temperature, humidity, people_count);
  
  // Calculate KPIv
  kpiv = calculateKPIv(carbon_value, people_count);
  person_co2 = (carbon_value - BASE_CO2) / CO2_PER_PERSON;
  if (person_co2 < 0) person_co2 = 0;

  // Display Temperature & Humidity
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.print(temperature);
  lcd.print("C  ");
    
  lcd.setCursor(0, 1);
  lcd.print("Humi: ");
  lcd.print(humidity);
  lcd.print("%   ");
  delay(1000);

  // Display CO2 Levels
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("CO2 Level: ");
  lcd.print(carbon_value);
  
  lcd.setCursor(0, 1);
  lcd.print("KPIv: ");
  lcd.print(kpiv, 2);
  lcd.print(kpiv < 1 ? " GOOD" : " POOR");
  delay(1000);
  
  // Display trend prediction
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Trend: ");
  if (trend > trend_threshold) {
    lcd.print("RISING");
  } else if (trend < -trend_threshold) {
    lcd.print("FALLING");
  } else {
    lcd.print("STABLE");
  }
  
  lcd.setCursor(0, 1);
  lcd.print("People: ");
  lcd.print(people_count);
  lcd.print("/");
  lcd.print(person_co2);
  delay(1000);

  // Format data to send to Raspberry Pi
  // Format: aTEMPbHUMIDITYcCO2dIN_COUNTeOUT_COUNTfKPIvgTRENDh
  uno = String("a") + String(temperature) + 
        String("b") + String(humidity) + 
        String("c") + String(carbon_value) + 
        String("d") + String(in_count) + 
        String("e") + String(out_count) + 
        String("f") + String(kpiv, 2) + 
        String("g") + String(trend, 2) + 
        String("h");
  
  Serial.println(uno);
  delay(1000);
  
  // Temperature Alert
  if (temperature >= TEMP_THRESHOLD) {
    digitalWrite(relay, LOW);
    digitalWrite(buzzer, HIGH);
    delay(5000);
    digitalWrite(relay, HIGH);
    digitalWrite(buzzer, LOW);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("High Temp Alert!");
    delay(1000);
  } else {
    digitalWrite(relay, HIGH);
    digitalWrite(buzzer, LOW);
  }

  // CO2 Alert & KPIv Alert
  if (carbon_value >= CO2_THRESHOLD || kpiv >= 1.0) {
    digitalWrite(buzzer, HIGH);
    lcd.clear();
    lcd.setCursor(0, 0);
    if (carbon_value >= CO2_THRESHOLD) {
      lcd.print("High CO2 Alert!");
    } else {
      lcd.print("Poor Ventilation!");
    }
    delay(1000);
  } else {
    digitalWrite(buzzer, LOW);
  }
}