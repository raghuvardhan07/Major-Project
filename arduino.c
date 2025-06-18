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

int in_sensor_pin = 8;   
int out_sensor_pin = 9;  
int co2_pin = A0;
int relay_pin = 11;
int buzzer_pin = 10;

// --- State Machine for People Counting ---
enum State { IDLE, IN_FIRST, OUT_FIRST, PASSING_IN, PASSING_OUT };
State currentState = IDLE;
int people_count = 0; // Net count of people inside

// Debounce variables
unsigned long lastDebounceTimeIn = 0;
unsigned long lastDebounceTimeOut = 0;
unsigned long debounceDelay = 50; // ms debounce time
int lastInState = LOW;
int lastOutState = LOW;
int stableInState = LOW;
int stableOutState = LOW;
// --- End State Machine Variables ---

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
  lcd.print("Prediction V2"); // Indicate updated version
  delay(2000);
  lcd.clear();

  pinMode(in_sensor_pin, INPUT);
  pinMode(out_sensor_pin, INPUT);
  pinMode(co2_pin, INPUT);
  pinMode(relay_pin, OUTPUT);
  pinMode(buzzer_pin, OUTPUT);

  digitalWrite(relay_pin, HIGH); // Relay is likely HIGH to be off
  digitalWrite(buzzer_pin, LOW);
  
  // Initialize buffers
  for(int i = 0; i < BUFFER_SIZE; i++) {
    co2_buffer[i] = 0;
    temp_buffer[i] = 0;
  }
}

// Function to read and debounce a sensor pin
int readDebounced(int pin, int& lastState, unsigned long& lastDebounceTime) {
  int reading = digitalRead(pin);
  if (reading != lastState) {
    lastDebounceTime = millis();
  }
  int stableState = lastState; // Assume current stable state
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading; // New stable state confirmed
    }
  }
  lastState = reading; // Update last read state
  return stableState;
}

// --- State Machine Logic for People Counting ---
void updateOccupancy() {
  // Read debounced sensor states (1 = beam broken, 0 = beam clear)
  int in_state = readDebounced(in_sensor_pin, lastInState, lastDebounceTimeIn);
  int out_state = readDebounced(out_sensor_pin, lastOutState, lastDebounceTimeOut);

  // If stable state has changed, update internal debounced state
  if(in_state != stableInState) stableInState = in_state;
  if(out_state != stableOutState) stableOutState = out_state;

  switch (currentState) {
    case IDLE:
      if (stableInState == HIGH && stableOutState == LOW) { // Inner beam broken first
        currentState = IN_FIRST;
      } else if (stableInState == LOW && stableOutState == HIGH) { // Outer beam broken first
        currentState = OUT_FIRST;
      }
      // If both break simultaneously, ignore for now or handle as error
      break;

    case IN_FIRST:
      if (stableInState == HIGH && stableOutState == HIGH) { // Outer beam also breaks
        currentState = PASSING_IN;
      } else if (stableInState == LOW && stableOutState == LOW) { // Went back out without crossing second beam
        currentState = IDLE;
      }
      // Stay in IN_FIRST if only inner beam is broken
      break;

    case OUT_FIRST:
      if (stableInState == HIGH && stableOutState == HIGH) { // Inner beam also breaks
        currentState = PASSING_OUT;
      } else if (stableInState == LOW && stableOutState == LOW) { // Went back without crossing second beam
        currentState = IDLE;
      }
       // Stay in OUT_FIRST if only outer beam is broken
      break;

    case PASSING_IN: // Both beams broken, started with inner
      if (stableInState == LOW && stableOutState == HIGH) { // Inner beam cleared, person entered
        people_count++;
        currentState = OUT_FIRST; // Now only outer is broken
        lcd.clear();
        lcd.print("Entered!");
        lcd.setCursor(0,1);
        lcd.print("Count: "); lcd.print(people_count);
        delay(500); // Short delay for message
      } else if (stableInState == LOW && stableOutState == LOW) { // Both cleared ~simultaneously
         currentState = IDLE; // Reset, missed exit trigger? Or fast pass? Count anyway.
         people_count++; 
         lcd.clear();
         lcd.print("Entered (Fast!)");
         lcd.setCursor(0,1);
         lcd.print("Count: "); lcd.print(people_count);
         delay(500);
      }
      // Stay in PASSING_IN if both are still broken
      break;

    case PASSING_OUT: // Both beams broken, started with outer
      if (stableInState == HIGH && stableOutState == LOW) { // Outer beam cleared, person exited
        if (people_count > 0) people_count--;
        currentState = IN_FIRST; // Now only inner is broken
        lcd.clear();
        lcd.print("Exited!");
        lcd.setCursor(0,1);
        lcd.print("Count: "); lcd.print(people_count);
        delay(500); // Short delay for message
      } else if (stableInState == LOW && stableOutState == LOW) { // Both cleared ~simultaneously
         currentState = IDLE; // Reset, missed exit trigger? Or fast pass? Count anyway.
         if (people_count > 0) people_count--;
         lcd.clear();
         lcd.print("Exited (Fast!)");
         lcd.setCursor(0,1);
         lcd.print("Count: "); lcd.print(people_count);
         delay(500);
      }
      // Stay in PASSING_OUT if both are still broken
      break;
  }

  // Add a timeout check: if stuck in a state for too long, reset to IDLE
  static unsigned long lastStateChangeTime = 0;
  if (millis() - lastStateChangeTime > 5000 && currentState != IDLE) { // 5 second timeout
     currentState = IDLE; // Reset if potentially stuck
     lastStateChangeTime = millis(); // Prevent immediate re-timeout
  } else if (currentState == IDLE){
     lastStateChangeTime = millis(); // Keep resetting timer while IDLE
  }

}
// --- End State Machine Logic ---

// Calculate moving average
int calculateMovingAverage(int buffer[], int size) {
  long sum = 0;
  int count = buffer_filled ? BUFFER_SIZE : buffer_index;
  if (count == 0) return 0; // Avoid division by zero
  
  for(int i = 0; i < count; i++) {
    sum += buffer[i];
  }
  return sum / count;
}

// Lightweight prediction model
float predictTrend(int current_co2, int current_temp, int current_humidity) {
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
                         (humidity_weight * (people_count > 0 ? 0.5 : -0.5)); // Humidity influence tied to occupancy
  
  return weighted_trend;
}

// Calculate KPIv (Ventilation KPI)
float calculateKPIv(int carbon_value, int actual_people) {
  if (carbon_value > ALARM_CO2) {
    return VN;
  }
  int estimated_people = (carbon_value - BASE_CO2) / CO2_PER_PERSON;
  if (estimated_people < 0) estimated_people = 0;
  
  if (actual_people > 0) {
    return (float)estimated_people / actual_people;
  } else {
     // If no actual people, but CO2 is above baseline, ventilation might be poor
     return (estimated_people > 0) ? 0.5 : 0.0; 
  }
}

// Check for model updates from Raspberry Pi
void checkForModelUpdates() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    if (data.startsWith("MODEL:")) {
      data.trim(); // Remove potential newline characters
      data = data.substring(6); // Remove "MODEL:"
      int start = 0;
      int end = data.indexOf(',');
      while (end != -1) {
        updateParameter(data.substring(start, end));
        start = end + 1;
        end = data.indexOf(',', start);
      }
      updateParameter(data.substring(start)); // Process last parameter
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
  if (colon == -1) return; // Invalid format
  String name = param.substring(0, colon);
  float value = param.substring(colon + 1).toFloat();
  
  if (name == "co2_weight") co2_weight = value;
  else if (name == "temp_weight") temp_weight = value;
  else if (name == "humidity_weight") humidity_weight = value;
  else if (name == "trend_threshold") trend_threshold = value;
}

unsigned long lastDisplayUpdate = 0;
int displayMode = 0; // 0: Temp/Humi, 1: CO2/KPIv, 2: Trend/People

void loop() {
  unsigned long currentTime = millis();
  
  // Non-blocking sensor updates and calculations
  checkForModelUpdates(); // Check serial first
  updateOccupancy();      // Update people count state machine

  // Read other sensors less frequently if needed, or keep as is
  int temperature = dht.readTemperature();
  int humidity = dht.readHumidity();
  // Check if reads failed and use last known value maybe? (DHT can be slow/fail)
  if (isnan(temperature) || isnan(humidity)) {
    // Handle error - maybe use previous values? For now, just skip calculation
    temperature = calculateMovingAverage(temp_buffer, BUFFER_SIZE); // Use average if read fails
    humidity = 50; // Default humidity if read fails
  } else {
    // Update buffers only on successful read
    temp_buffer[buffer_index] = temperature;
  }

  int carbon_value = analogRead(co2_pin);
  co2_buffer[buffer_index] = carbon_value; // Update CO2 buffer regardless
  
  buffer_index = (buffer_index + 1) % BUFFER_SIZE;
  if (buffer_index == 0 && !buffer_filled) buffer_filled = true;

  // Calculations
  float trend = predictTrend(carbon_value, temperature, humidity);
  kpiv = calculateKPIv(carbon_value, people_count);
  person_co2 = max(0, (carbon_value - BASE_CO2) / CO2_PER_PERSON);

  // Non-blocking Display Updates (Cycle every 2 seconds)
  if (currentTime - lastDisplayUpdate > 2000) {
     lcd.clear();
     switch (displayMode) {
        case 0: // Temp/Humi
          lcd.setCursor(0, 0);
          lcd.print("Temp: "); lcd.print(temperature); lcd.print("C");
          lcd.setCursor(0, 1);
          lcd.print("Humi: "); lcd.print(humidity); lcd.print("%");
          break;
        case 1: // CO2/KPIv
          lcd.setCursor(0, 0);
          lcd.print("CO2: "); lcd.print(carbon_value); lcd.print("ppm");
          lcd.setCursor(0, 1);
          lcd.print("KPIv: "); lcd.print(kpiv, 2);
          lcd.print(kpiv < 1 ? " G" : " P"); // Good/Poor
          break;
        case 2: // Trend/People
           lcd.setCursor(0, 0);
           lcd.print("Trend: ");
           if (trend > trend_threshold) lcd.print("Rising");
           else if (trend < -trend_threshold) lcd.print("Falling");
           else lcd.print("Stable");
           lcd.setCursor(0, 1);
           lcd.print("People: "); lcd.print(people_count);
           lcd.print("/"); lcd.print(person_co2); // Actual / CO2 Estimated
           break;
     }
     displayMode = (displayMode + 1) % 3; // Cycle through 3 display modes
     lastDisplayUpdate = currentTime;
  }

  // Send data to Raspberry Pi (maybe less frequently?)
  static unsigned long lastSerialSend = 0;
  if (currentTime - lastSerialSend > 5000) { // Send every 5 seconds
      uno = String("a") + String(temperature) + 
            String("b") + String(humidity) + 
            String("c") + String(carbon_value) + 
            String("d") + String(people_count) + // Send corrected people_count
            String("e") + String(0) + // Sending 0 for out_count, as it's not used for occupancy
            String("f") + String(kpiv, 2) + 
            String("g") + String(trend, 2) + 
            String("h");
      Serial.println(uno);
      lastSerialSend = currentTime;
  }
  
  // Alerts (should be non-blocking if possible)
  static bool highTempAlertActive = false;
  static bool co2AlertActive = false;
  static unsigned long alertStartTime = 0;

  // Temperature Alert Logic
  if (temperature >= TEMP_THRESHOLD && !highTempAlertActive) {
    digitalWrite(relay_pin, LOW); // Turn ON Fan
    digitalWrite(buzzer_pin, HIGH);
    highTempAlertActive = true;
    alertStartTime = currentTime;
    // Maybe display alert message immediately here
    lcd.clear(); lcd.print("High Temp Alert!"); 
    lastDisplayUpdate = currentTime; // Reset display timer
    displayMode = 0; // Force display back to temp/humi
  } else if (highTempAlertActive && (currentTime - alertStartTime > 5000)) { // Alert duration 5s
    digitalWrite(relay_pin, HIGH); // Turn OFF Fan
    digitalWrite(buzzer_pin, LOW);
    highTempAlertActive = false;
  } else if (temperature < TEMP_THRESHOLD && highTempAlertActive) { // Condition cleared early
     digitalWrite(relay_pin, HIGH); 
     digitalWrite(buzzer_pin, LOW);
     highTempAlertActive = false;
  }

  // CO2/KPIv Alert Logic (only buzz if temp alert isn't active)
  if ((carbon_value >= CO2_THRESHOLD || kpiv >= 1.0) && !co2AlertActive && !highTempAlertActive) {
     digitalWrite(buzzer_pin, HIGH);
     co2AlertActive = true;
     // Maybe display alert message immediately here
     lcd.clear(); 
     if(carbon_value >= CO2_THRESHOLD) lcd.print("High CO2 Alert!");
     else lcd.print("Poor Vent Alert!");
     lastDisplayUpdate = currentTime; // Reset display timer
     displayMode = 1; // Force display back to co2/kpiv
  } else if (co2AlertActive && !(carbon_value >= CO2_THRESHOLD || kpiv >= 1.0)) { // Condition cleared
     digitalWrite(buzzer_pin, LOW);
     co2AlertActive = false;
  } else if (co2AlertActive && highTempAlertActive) { // Temp alert takes precedence for buzzer
      digitalWrite(buzzer_pin, LOW); // Ensure buzzer is off if temp alert is on
      // Keep co2AlertActive = true, so buzzer resumes if temp alert clears first
  }
}