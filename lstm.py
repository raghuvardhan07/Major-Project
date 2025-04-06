import serial
import time
import numpy as np
import os
import json
import requests
import threading
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classroom_monitor.log"),
        logging.StreamHandler()
    ]
)

# LCD Pin Configuration
LCD_RS = 25
LCD_EN = 24
LCD_D4 = 23
LCD_D5 = 17
LCD_D6 = 18
LCD_D7 = 22

# Initialize LCD
lcd = CharLCD(
    pin_rs=LCD_RS,
    pin_e=LCD_EN,
    pins_data=[LCD_D4, LCD_D5, LCD_D6, LCD_D7],
    numbering_mode=GPIO.BCM,
    cols=16,
    rows=2
)

# Classroom metadata - can be expanded
CLASSROOM_METADATA = {
    'classroom1': {
        'room_size': 50,  # square meters
        'capacity': 25,   # maximum people
        'ventilation_type': 'natural',
        'window_area': 8  # square meters
    },
    'classroom2': {
        'room_size': 60,
        'capacity': 30,
        'ventilation_type': 'hvac',
        'window_area': 6
    }
}

# Path for model storage
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Configuration
CLOUD_UPLOAD_INTERVAL = 3600  # seconds (1 hour)
MODEL_UPDATE_INTERVAL = 86400  # seconds (24 hours)
DATA_HARMONIZATION_WINDOW = 5  # readings to use for harmonization

# Serial Port Configuration
usb0_port = "/dev/ttyS0"
usb1_port = "/dev/ttyS1"
baud_rate = 9600


class DataHarmonizer:
    """Harmonizes raw data from sensors"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.data_buffers = {}
        self.scalers = {}
    
    def add_data_point(self, classroom_id, data_point):
        """Add a new data point to the buffer"""
        if classroom_id not in self.data_buffers:
            self.data_buffers[classroom_id] = []
            self.scalers[classroom_id] = MinMaxScaler()
        
        self.data_buffers[classroom_id].append(data_point)
        
        # Keep buffer at window size
        if len(self.data_buffers[classroom_id]) > self.window_size:
            self.data_buffers[classroom_id].pop(0)
    
    def harmonize(self, classroom_id, data_point):
        """Harmonize a data point based on historical data"""
        self.add_data_point(classroom_id, data_point)
        
        # Not enough data for harmonization
        if len(self.data_buffers[classroom_id]) < 2:
            return data_point
        
        # Convert buffer to numpy array
        data_array = np.array(self.data_buffers[classroom_id])
        
        # Check for outliers (Z-score)
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0)
        
        # Replace outliers with moving average
        harmonized_point = data_point.copy()
        for i in range(len(data_point)):
            if stds[i] > 0:  # Avoid division by zero
                z_score = abs((data_point[i] - means[i]) / stds[i])
                if z_score > 3:  # Outlier threshold
                    harmonized_point[i] = means[i]
        
        # Add classroom context
        harmonized_point = self.add_context(harmonized_point, classroom_id)
        
        return harmonized_point
    
    def add_context(self, data_point, classroom_id):
        """Add classroom-specific context to data"""
        if classroom_id in CLASSROOM_METADATA:
            metadata = CLASSROOM_METADATA[classroom_id]
            
            # Normalize occupancy by capacity
            if len(data_point) >= 5:  # Make sure we have occupancy data
                occupancy = data_point[3]  # inside_count
                capacity = metadata['capacity']
                data_point[3] = min(1.0, occupancy / capacity) if capacity > 0 else 0
                
            # Add timestamp as a feature (hour of day normalized)
            hour = datetime.now().hour / 24.0
            data_point = np.append(data_point, hour)
            
        return data_point

class LSTMModel:
    """Advanced LSTM model with pattern learning"""
    
    def __init__(self, classroom_id, input_shape=(1, 8), cloud_sync=True):
        self.classroom_id = classroom_id
        self.model_path = os.path.join(MODEL_DIR, f"{classroom_id}_model.h5")
        self.scaler = MinMaxScaler()
        self.cloud_sync = cloud_sync
        self.last_update = time.time()
        self.training_data = []
        self.input_shape = input_shape
        self.model = self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one if none exists"""
        if os.path.exists(self.model_path):
            logging.info(f"Loading existing model for {self.classroom_id}")
            try:
                return load_model(self.model_path)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        
        logging.info(f"Creating new model for {self.classroom_id}")
        return self.create_model()
    
    def create_model(self):
        """Create a new LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5)  # Output: temp, humidity, co2, occupancy, KPIv
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def add_training_data(self, data):
        """Add data for training"""
        self.training_data.append(data)
        
        # Keep only recent data
        max_samples = 1000
        if len(self.training_data) > max_samples:
            self.training_data = self.training_data[-max_samples:]
    
    def should_update(self):
        """Check if model should be updated"""
        return (time.time() - self.last_update) >= MODEL_UPDATE_INTERVAL
    
    def train(self, force=False):
        """Train or update the model with collected data"""
        if not force and not self.should_update():
            return False
            
        if len(self.training_data) < 30:  # Need enough data
            logging.info(f"Not enough training data for {self.classroom_id}")
            return False
            
        logging.info(f"Training model for {self.classroom_id}")
        
        # Prepare data
        data = np.array(self.training_data)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - 10):
            X.append(scaled_data[i:i+10])
            y.append(scaled_data[i+10, :5])  # First 5 features are the target
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        
        # Save model
        self.model.save(self.model_path)
        self.last_update = time.time()
        
        # Sync with cloud if enabled
        if self.cloud_sync:
            self.sync_with_cloud()
            
        return True
    
    def predict(self, data_sequence, steps=10):
        """Make predictions for future values"""
        if len(data_sequence) < 10:
            # Pad with zeros if not enough data
            padding = np.zeros((10 - len(data_sequence), data_sequence.shape[1]))
            data_sequence = np.vstack((padding, data_sequence))
        
        # Use last 10 points
        data_sequence = data_sequence[-10:]
        
        # Scale data
        scaled_sequence = self.scaler.transform(data_sequence)
        
        # Reshape for prediction
        X = scaled_sequence.reshape(1, 10, scaled_sequence.shape[1])
        
        # Make predictions
        predictions = []
        current_sequence = X
        
        for _ in range(steps):
            # Predict next value
            next_val = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_val[0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :5] = next_val[0]
        
        # Inverse transform predictions
        padded_predictions = np.zeros((len(predictions), scaled_sequence.shape[1]))
        padded_predictions[:, :5] = predictions  # First 5 values are our predictions
        
        return self.scaler.inverse_transform(padded_predictions)[:, :5]
    
    def sync_with_cloud(self):
        """Sync model with cloud"""
        try:
            # Check if newer model exists in cloud
            cloud_model_url = f"https://your-cloud-storage.com/models/{self.classroom_id}_model.h5"
            cloud_model_info_url = f"https://your-cloud-storage.com/models/{self.classroom_id}_info.json"
            
            # Get cloud model info
            response = requests.get(cloud_model_info_url)
            if response.status_code == 200:
                cloud_info = response.json()
                cloud_update_time = cloud_info.get('last_update', 0)
                
                # If cloud model is newer, download it
                if cloud_update_time > self.last_update:
                    logging.info(f"Downloading newer model from cloud for {self.classroom_id}")
                    model_response = requests.get(cloud_model_url)
                    if model_response.status_code == 200:
                        with open(self.model_path, 'wb') as f:
                            f.write(model_response.content)
                        self.model = load_model(self.model_path)
                        self.last_update = cloud_update_time
                        return True
            
            # Upload current model to cloud
            # (This would be implemented based on your cloud provider)
            logging.info(f"Uploading model to cloud for {self.classroom_id}")
            
            return True
        except Exception as e:
            logging.error(f"Error syncing with cloud: {e}")
            return False

class ThingSpeakManager:
    def __init__(self, api_keys):
        """
        Initialize with ThingSpeak API keys
        api_keys: dict with classroom_id as key and write_key as value
        """
        self.api_keys = api_keys
        self.base_url = "https://api.thingspeak.com/update"
        self.last_upload_time = {}
        self.upload_interval = 15  # ThingSpeak free account requires 15 sec between updates
    
    def upload_data(self, classroom_id, data):
        """Upload classroom data to ThingSpeak"""
        # Check if enough time has passed since last upload
        current_time = time.time()
        if classroom_id in self.last_upload_time and \
           (current_time - self.last_upload_time[classroom_id]) < self.upload_interval:
            logging.info(f"Skipping upload - need to wait {self.upload_interval}s between updates")
            return False
        
        # Extract values from data
        temperature = data[0]
        humidity = data[1]
        co2 = data[2]
        occupancy = data[3]  # People count
        kpiv = data[5] if len(data) > 5 else 0  # KPIv value
        trend = data[6] if len(data) > 6 else 0  # Trend prediction
        
        # Calculate alert status
        alert_status = 0
        if kpiv >= 1.0 or co2 > 1000:
            alert_status = 2  # Critical
        elif kpiv >= 0.8 or co2 > 800:
            alert_status = 1  # Warning
            
        # Get API key for this classroom
        api_key = self.api_keys.get(classroom_id)
        if not api_key:
            logging.warning(f"No API key found for {classroom_id}")
            return False
            
        # Build payload
        payload = {
            'api_key': api_key,
            'field1': temperature,
            'field2': humidity,
            'field3': co2,
            'field4': occupancy,
            'field5': kpiv,
            'field6': trend,
            'field7': alert_status,
            'field8': 1.0  # Model version
        }
        
        try:
            # Send data to ThingSpeak
            response = requests.get(self.base_url, params=payload)
            if response.status_code == 200:
                self.last_upload_time[classroom_id] = current_time
                logging.info(f"Data uploaded to ThingSpeak for {classroom_id}")
                return True
            else:
                logging.error(f"Error uploading to ThingSpeak: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Exception during ThingSpeak upload: {e}")
            return False
    
    def upload_batch(self, classroom_data):
        """Upload latest data for all classrooms"""
        results = {}
        for classroom_id, data_list in classroom_data.items():
            if data_list:
                # Get latest reading
                latest_data = data_list[-1]
                results[classroom_id] = self.upload_data(classroom_id, latest_data)
                
                # ThingSpeak rate limiting - wait between uploads for different channels
                time.sleep(16)  # Wait a bit more than the minimum required time
        
        return results

class ClassroomMonitor:
    """Main class for monitoring classrooms"""
    
    def __init__(self):
        # Initialize serial connections
        try:
            self.ser_usb0 = serial.Serial(usb0_port, baud_rate, timeout=1)
            self.ser_usb1 = serial.Serial(usb1_port, baud_rate, timeout=1)
            logging.info("Serial connections established")
        except Exception as e:
            logging.error(f"Error opening serial ports: {e}")
            raise
        
        # Initialize components
        self.harmonizer = DataHarmonizer(window_size=DATA_HARMONIZATION_WINDOW)
        self.models = {
            'classroom1': LSTMModel('classroom1'),
            'classroom2': LSTMModel('classroom2')
        }
        
        # Data storage
        self.classroom_data = {
            'classroom1': [],
            'classroom2': []
        }
        
        self.last_cloud_upload = time.time()
        
        # Display welcome message
        lcd.clear()
        lcd.write_string("Classroom")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Monitor v2.0")
        time.sleep(2)
        
        # ThingSpeak API keys - replace with your actual keys
        thingspeak_api_keys = {
            'classroom1': 'FUTC7Y9NKV47NROU',
            'classroom2': 'K8GEMD0ZWCWSU4SB'
        }
        self.thingspeak = ThingSpeakManager(thingspeak_api_keys)
        
        # Set upload interval (default 15min, frequent enough for monitoring)
        self.cloud_upload_interval = 900  # 15 minutes in seconds
    
    def parse_data(self, data):
        """Parse data from serial connection"""
        try:
            temperature = int(data[data.index('a') + 1:data.index('b')])
            humidity = int(data[data.index('b') + 1:data.index('c')])
            co2 = int(data[data.index('c') + 1:data.index('d')])
            inside_count = int(data[data.index('d') + 1:data.index('e')])
            outside_count = int(data[data.index('e') + 1:data.index('f')])
            
            # Parse KPIv if available
            kpiv = 0.0
            if 'f' in data and 'g' in data:
                kpiv = float(data[data.index('f') + 1:data.index('g')])
                
            # Parse trend if available
            trend = 0.0
            if 'g' in data and 'h' in data:
                trend = float(data[data.index('g') + 1:data.index('h')])
                
            return [temperature, humidity, co2, inside_count, outside_count, kpiv, trend]
        except Exception as e:
            logging.error(f"Error parsing data: {e}, data: {data}")
            return None
    
    def collect_data(self, duration=60):
        """Collect data from both classrooms for specified duration"""
        logging.info(f"Collecting data for {duration} seconds")
        
        start_time = time.time()
        classroom1_count = 0
        classroom2_count = 0
        
        lcd.clear()
        lcd.write_string("Collecting data")
        
        while time.time() - start_time < duration:
            # Check classroom 1 (USB0)
            if self.ser_usb0.in_waiting > 0:
                data = self.ser_usb0.readline().decode().strip()
                if data.startswith("a") and ('f' in data):
                    parsed = self.parse_data(data)
                    if parsed:
                        # Harmonize data
                        harmonized = self.harmonizer.harmonize('classroom1', parsed)
                        self.classroom_data['classroom1'].append(harmonized)
                        self.models['classroom1'].add_training_data(harmonized)
                        classroom1_count += 1
                        logging.info(f"Classroom 1 reading: {parsed}")
            
            # Check classroom 2 (USB1)
            if self.ser_usb1.in_waiting > 0:
                data = self.ser_usb1.readline().decode().strip()
                if data.startswith("a") and ('f' in data):
                    parsed = self.parse_data(data)
                    if parsed:
                        # Harmonize data
                        harmonized = self.harmonizer.harmonize('classroom2', parsed)
                        self.classroom_data['classroom2'].append(harmonized)
                        self.models['classroom2'].add_training_data(harmonized)
                        classroom2_count += 1
                        logging.info(f"Classroom 2 reading: {parsed}")
            
            # Update LCD
            if time.time() % 5 < 0.1:  # Update every ~5 seconds
                lcd.clear()
                lcd.write_string(f"C1 Readings: {classroom1_count}")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"C2 Readings: {classroom2_count}")
            
            # Upload to cloud if interval elapsed
            if time.time() - self.last_cloud_upload >= self.cloud_upload_interval:
                self.thingspeak.upload_batch(self.classroom_data)
                self.last_cloud_upload = time.time()
            
            time.sleep(0.1)
        
        logging.info(f"Collected {classroom1_count} readings from Classroom 1")
        logging.info(f"Collected {classroom2_count} readings from Classroom 2")
        
        return classroom1_count > 0 and classroom2_count > 0
    
    def send_model_updates(self, classroom_id):
        """Send model parameter updates to Arduino"""
        serial_conn = self.ser_usb0 if classroom_id == 'classroom1' else self.ser_usb1
        
        # Generate parameters based on recent data
        params = self.calculate_model_parameters(classroom_id)
        
        command = "MODEL:"
        for key, value in params.items():
            command += f"{key}:{value},"
        command = command[:-1]  # Remove trailing comma
        
        try:
            serial_conn.write((command + "\n").encode())
            logging.info(f"Sent model update to {classroom_id}: {command}")
            return True
        except Exception as e:
            logging.error(f"Error sending model update: {e}")
            return False
    
    def calculate_model_parameters(self, classroom_id):
        """Calculate model parameters based on recent data"""
        recent_data = self.classroom_data.get(classroom_id, [])
        
        if len(recent_data) < 10:
            # Default parameters if not enough data
            return {
                "co2_weight": 0.6,
                "temp_weight": 0.3,
                "humidity_weight": 0.1,
                "trend_threshold": 0.7
            }
        
        # Analyze recent data to determine weights
        # This is a simplified approach - you would use more sophisticated methods
        recent_data = np.array(recent_data[-50:])
        
        # Calculate correlations with KPIv
        kpiv_idx = 5  # Index of KPIv in data
        co2_corr = np.abs(np.corrcoef(recent_data[:, 2], recent_data[:, kpiv_idx])[0, 1])
        temp_corr = np.abs(np.corrcoef(recent_data[:, 0], recent_data[:, kpiv_idx])[0, 1])
        humidity_corr = np.abs(np.corrcoef(recent_data[:, 1], recent_data[:, kpiv_idx])[0, 1])
        
        # Handle NaN
        co2_corr = 0.6 if np.isnan(co2_corr) else co2_corr
        temp_corr = 0.3 if np.isnan(temp_corr) else temp_corr
        humidity_corr = 0.1 if np.isnan(humidity_corr) else humidity_corr
        
        # Normalize weights
        total = co2_corr + temp_corr + humidity_corr
        if total > 0:
            co2_weight = co2_corr / total
            temp_weight = temp_corr / total
            humidity_weight = humidity_corr / total
        else:
            co2_weight = 0.6
            temp_weight = 0.3
            humidity_weight = 0.1
        
        # Calculate trend threshold based on data variance
        variances = np.var(recent_data[:, :3], axis=0)
        avg_variance = np.mean(variances)
        trend_threshold = 0.5 + (0.5 * (1 - np.exp(-avg_variance / 100)))
        
        return {
            "co2_weight": round(co2_weight, 2),
            "temp_weight": round(temp_weight, 2),
            "humidity_weight": round(humidity_weight, 2),
            "trend_threshold": round(trend_threshold, 2)
        }
    
    def train_models(self):
        """Train all models"""
        for classroom_id, model in self.models.items():
            if model.train():
                logging.info(f"Trained model for {classroom_id}")
                # Send model updates to Arduino
                self.send_model_updates(classroom_id)
    
    def make_predictions(self, steps=10):
        """Make predictions for both classrooms"""
        predictions = {}
        
        for classroom_id, model in self.models.items():
            if len(self.classroom_data[classroom_id]) >= 10:
                data = np.array(self.classroom_data[classroom_id][-10:])
                pred = model.predict(data, steps=steps)
                predictions[classroom_id] = pred
                logging.info(f"Made predictions for {classroom_id}")
        
        return predictions
    
    def score_predictions(self, predictions):
        """Score predictions to determine best classroom"""
        scores = {}
        
        for classroom_id, pred in predictions.items():
            score = 0
            for p in pred:
                temp, hum, co2, occupancy, kpiv = p[:5]
                
                # Weight kpiv most heavily
                score += kpiv * 5
                
                # Temperature deviation from ideal (24°C)
                score += abs(temp - 24) * 0.2
                
                # CO2 penalty
                score += (co2 / 1000) * 0.3
                
                # Occupancy penalty
                score += occupancy * 0.1
            
            scores[classroom_id] = score
            logging.info(f"{classroom_id} score: {score}")
        
        return scores
    
    def get_recommendation(self):
        """Get classroom recommendation based on predictions"""
        # Collect a batch of data
        if not self.collect_data(duration=60):
            return "Not enough data"
        
        # Train models if needed
        self.train_models()
        
        # Make predictions
        predictions = self.make_predictions(steps=10)
        if len(predictions) < 2:
            return "Not enough data for prediction"
        
        # Score predictions
        scores = self.score_predictions(predictions)
        
        # Determine best classroom
        if scores['classroom1'] < scores['classroom2']:
            recommendation = "Classroom 1"
            details = f"Score: {scores['classroom1']:.2f} vs {scores['classroom2']:.2f}"
        elif scores['classroom2'] < scores['classroom1']:
            recommendation = "Classroom 2"
            details = f"Score: {scores['classroom2']:.2f} vs {scores['classroom1']:.2f}"
else:
            recommendation = "Both Equal"
            details = f"Score: {scores['classroom1']:.2f}"
        
        logging.info(f"Recommendation: {recommendation}, {details}")
        return recommendation, details
    
    def run(self):
        """Main run loop"""
        try:
            while True:
                # Get recommendation
                result = self.get_recommendation()
                
                if isinstance(result, tuple):
                    recommendation, details = result
                else:
                    recommendation = result
                    details = ""
                
                # Display on LCD
lcd.clear()
                lcd.write_string("Better Room:")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(recommendation)
                
                logging.info(f"Recommended: {recommendation}")
                
                # Upload data to ThingSpeak
                if time.time() - self.last_cloud_upload >= self.cloud_upload_interval:
                    self.thingspeak.upload_batch(self.classroom_data)
                    self.last_cloud_upload = time.time()
                
                # Wait before next iteration
                time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            logging.info("Program terminated by user")
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            # Clean up
            lcd.clear()
            GPIO.cleanup()

if __name__ == "__main__":
    monitor = ClassroomMonitor()
    monitor.run()