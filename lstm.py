import serial
import time
import numpy as np
import os
import json
import requests
import threading
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("classroom_monitor.log"), logging.StreamHandler()])

LCD_RS, LCD_EN, LCD_D4, LCD_D5, LCD_D6, LCD_D7 = 25, 24, 23, 17, 18, 22
lcd = CharLCD(pin_rs=LCD_RS, pin_e=LCD_EN, pins_data=[LCD_D4, LCD_D5, LCD_D6, LCD_D7],
    numbering_mode=GPIO.BCM, cols=16, rows=2)

CLOUD_UPLOAD_INTERVAL = 3600
MODEL_UPDATE_INTERVAL = 86400
DATA_HARMONIZATION_WINDOW = 5

CLASSROOM_METADATA = {
    'classroom1': {'room_size': 50, 'capacity': 25, 'ventilation_type': 'natural', 'window_area': 8},
    'classroom2': {'room_size': 60, 'capacity': 30, 'ventilation_type': 'hvac', 'window_area': 6}
}

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

PRETRAINED_WEIGHTS_PATH = 'pretrained_lstm.weights.h5'
PRETRAINED_INPUT_SHAPE = (10, 5) 
FINE_TUNE_LEARNING_RATE = 1e-4 

usb0_port, usb1_port, baud_rate = "/dev/ttyS0", "/dev/ttyS1", 9600

def calculate_kpiv(co2, occupancy, base_co2=400, co2_per_person=20, alarm_co2=1000, vn=2.0):
    if co2 > alarm_co2:
        return vn
    estimated_people = max(0, (co2 - base_co2) / co2_per_person)
    if occupancy > 0:
        return estimated_people / occupancy
    else:
        return 0.5 if estimated_people > 0 else 0.0

class DataHarmonizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.data_buffers = {}
        self.scalers = {}

    def add_data_point(self, classroom_id, data_point):
        if classroom_id not in self.data_buffers:
            self.data_buffers[classroom_id] = []
            self.scalers[classroom_id] = MinMaxScaler()
        self.data_buffers[classroom_id].append(data_point)
        if len(self.data_buffers[classroom_id]) > self.window_size:
            self.data_buffers[classroom_id].pop(0)

    def harmonize(self, classroom_id, data_point):
        self.add_data_point(classroom_id, data_point)
        if len(self.data_buffers[classroom_id]) < 2:
            return data_point
        data_array = np.array(self.data_buffers[classroom_id])
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0)
        harmonized_point = data_point.copy()
        for i in range(len(data_point)):
            if stds[i] > 0:  
                z_score = abs((data_point[i] - means[i]) / stds[i])
                if z_score > 3:  
                    harmonized_point[i] = means[i]
        return self.add_context(harmonized_point, classroom_id)

    def add_context(self, data_point, classroom_id):
        if classroom_id in CLASSROOM_METADATA:
            metadata = CLASSROOM_METADATA[classroom_id]
            if len(data_point) >= 5:  
                occupancy = data_point[3]  
                capacity = metadata['capacity']
                data_point[3] = min(1.0, occupancy / capacity) if capacity > 0 else 0
            hour = datetime.now().hour / 24.0
            data_point = np.append(data_point, hour)
        return data_point

class LSTMModel:
    def __init__(self, classroom_id, input_shape=(10, 8), cloud_sync=True):
        self.classroom_id = classroom_id
        self.model_path = os.path.join(MODEL_DIR, f"{classroom_id}_model.h5")
        self.scaler = MinMaxScaler()
        self.training_data = []
        self.input_shape = input_shape
        self.model = self.load_or_create_model()
        self.cloud_sync = cloud_sync
        self.last_update = time.time()
        self._pretrained_weights_loaded = False 
        self.performance_metrics = {}

    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                model = load_model(self.model_path)
                if len(self.training_data) >= 2:
                    try:
                        feature_size = self.input_shape[1]
                        valid_data = [item for item in self.training_data if isinstance(item, (list, np.ndarray)) and len(item) == feature_size]
                        if len(valid_data) >= 2:
                             self.scaler.fit(np.array(valid_data))
                    except Exception as fit_err:
                        logging.warning(f"Could not fit scaler after model load: {fit_err}")
                return model
            except Exception as e:
                logging.error(f"Error loading model: {e}")

        target_model = self.create_model() 
        if os.path.exists(PRETRAINED_WEIGHTS_PATH):
            try:
                pretrain_model_temp = Sequential([
                    LSTM(64, return_sequences=True, input_shape=PRETRAINED_INPUT_SHAPE, name='lstm_shared_1'),
                    Dropout(0.2, name='dropout_shared_1'),
                    LSTM(32, name='lstm_shared_2'),
                    Dropout(0.2, name='dropout_shared_2'),
                    Dense(16, activation='relu', name='dense_shared_1'),
                    Dense(PRETRAINED_INPUT_SHAPE[1], name='dense_pretrain_output') 
                ], name="PretrainLoader")

                pretrain_model_temp.load_weights(PRETRAINED_WEIGHTS_PATH)
                loaded_count = 0
                for layer in pretrain_model_temp.layers:
                    try:
                        target_layer = target_model.get_layer(layer.name)
                        if target_layer:
                            pretrain_weights = layer.get_weights()
                            target_weights = target_layer.get_weights()
                            if all(w1.shape == w2.shape for w1, w2 in zip(pretrain_weights, target_weights)):
                                target_layer.set_weights(pretrain_weights)
                                loaded_count += 1
                    except Exception:
                        pass
                if loaded_count > 0:
                     self._pretrained_weights_loaded = True 
            except Exception as e:
                logging.error(f"Could not load weights: {e}")
        return target_model 

    def create_model(self):
        timesteps, features = self.input_shape
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(timesteps, features), name='lstm_shared_1'),
            Dropout(0.2, name='dropout_shared_1'),
            LSTM(32, name='lstm_shared_2'),
            Dropout(0.2, name='dropout_shared_2'),
            Dense(16, activation='relu', name='dense_shared_1'),
            Dense(5, name='dense_target_output') 
        ], name=f"ClassroomLSTM_{self.classroom_id}")
        return model

    def add_training_data(self, data):
        if isinstance(data, np.ndarray):
             self.training_data.append(data.tolist()) 
        elif isinstance(data, list):
             self.training_data.append(data)
        else:
             return
        max_samples = 1000
        if len(self.training_data) > max_samples:
            self.training_data = self.training_data[-max_samples:]

    def _is_scaler_fitted(self):
         return hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None

    def should_update(self):
        return (time.time() - self.last_update) >= MODEL_UPDATE_INTERVAL

    def train(self, force=False):
        if not force and not self.should_update():
            return False

        if len(self.training_data) < 30:
            return False

        try:
            data = np.array(self.training_data)
            if data.ndim != 2:
                return False
            expected_features = self.input_shape[1]
            if data.shape[1] != expected_features:
                return False

            if not self._is_scaler_fitted():
                 if len(data) >= 2:
                      self.scaler.fit(data)
                 else:
                      return False
            scaled_data = self.scaler.transform(data)
        except Exception:
            return False

        X, y = [], []
        n_steps = self.input_shape[0]
        for i in range(len(scaled_data) - n_steps):
            X.append(scaled_data[i:i+n_steps])
            y.append(scaled_data[i+n_steps, :5])  

        if not X or not y:
             return False

        X, y = np.array(X), np.array(y)
        train_split = int(0.8 * len(X))
        X_train, X_val = X[:train_split], X[train_split:]
        y_train, y_val = y[:train_split], y[train_split:]

        if self.model is None:
             return False

        current_lr = FINE_TUNE_LEARNING_RATE if self._pretrained_weights_loaded else 0.001 
        self.model.compile(optimizer=Adam(learning_rate=current_lr), loss='mse', metrics=['mae'])

        try:
            history = self.model.fit(
                 X_train, y_train,
                 epochs=50, 
                 batch_size=32,
                 verbose=0, 
                 validation_data=(X_val, y_val),
                 callbacks=[
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), 
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7)
                 ]
            )
            self.performance_metrics = self.evaluate_performance(X_val, y_val)
        except Exception:
            return False

        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            self.last_update = time.time()
        except Exception:
            pass

        if self.cloud_sync:
            self.sync_with_cloud() 
        return True

    def evaluate_performance(self, X_val, y_val):
        metrics = {}
        y_pred_scaled = self.model.predict(X_val, verbose=0)
        mse = np.mean(np.square(y_val - y_pred_scaled))
        metrics["RMSE"] = np.sqrt(mse)
        metrics["MAE"] = np.mean(np.abs(y_val - y_pred_scaled))
        ss_total = np.sum(np.square(y_val - np.mean(y_val, axis=0)))
        ss_residual = np.sum(np.square(y_val - y_pred_scaled))
        metrics["R_squared"] = 1 - (ss_residual / ss_total)
        
        batch_size = y_val.shape[0]
        feature_size = self.input_shape[1]
        dummy_y_val_full = np.zeros((batch_size, feature_size))
        dummy_y_pred_full = np.zeros((batch_size, feature_size))
        dummy_y_val_full[:, :y_val.shape[1]] = y_val
        dummy_y_pred_full[:, :y_pred_scaled.shape[1]] = y_pred_scaled
        
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            for i in range(y_val.shape[1], feature_size):
                dummy_y_val_full[:, i] = self.scaler.mean_[i]
                dummy_y_pred_full[:, i] = self.scaler.mean_[i]
                
        y_val_original = self.scaler.inverse_transform(dummy_y_val_full)[:, :y_val.shape[1]]
        y_pred_original = self.scaler.inverse_transform(dummy_y_pred_full)[:, :y_pred_scaled.shape[1]]
        
        co2_idx = 2
        if y_val.shape[1] > co2_idx:
            co2_threshold_moderate = 800
            y_val_co2_moderate = (y_val_original[:, co2_idx] > co2_threshold_moderate).astype(int)
            y_pred_co2_moderate = (y_pred_original[:, co2_idx] > co2_threshold_moderate).astype(int)
            metrics["CO2_Moderate_Threshold_Accuracy"] = np.mean(y_val_co2_moderate == y_pred_co2_moderate)
            
            co2_threshold_critical = 1000
            y_val_co2_critical = (y_val_original[:, co2_idx] > co2_threshold_critical).astype(int)
            y_pred_co2_critical = (y_pred_original[:, co2_idx] > co2_threshold_critical).astype(int)
            metrics["CO2_Critical_Threshold_Accuracy"] = np.mean(y_val_co2_critical == y_pred_co2_critical)
            
            if y_val.shape[1] > 3:
                occ_idx = 3
                kpiv_val = np.array([calculate_kpiv(co2, occ) for co2, occ in 
                                 zip(y_val_original[:, co2_idx], y_val_original[:, occ_idx])])
                kpiv_pred = np.array([calculate_kpiv(co2, occ) for co2, occ in 
                                  zip(y_pred_original[:, co2_idx], y_pred_original[:, occ_idx])])
                
                kpiv_threshold_moderate = 0.8
                kpiv_val_moderate = (kpiv_val > kpiv_threshold_moderate).astype(int)
                kpiv_pred_moderate = (kpiv_pred > kpiv_threshold_moderate).astype(int)
                metrics["KPIv_Moderate_Threshold_Accuracy"] = np.mean(kpiv_val_moderate == kpiv_pred_moderate)
                
                kpiv_threshold_critical = 1.0
                kpiv_val_critical = (kpiv_val > kpiv_threshold_critical).astype(int)
                kpiv_pred_critical = (kpiv_pred > kpiv_threshold_critical).astype(int)
                metrics["KPIv_Critical_Threshold_Accuracy"] = np.mean(kpiv_val_critical == kpiv_pred_critical)
                metrics["KPIv_MAE"] = np.mean(np.abs(kpiv_val - kpiv_pred))
                
        feature_names = ["Temperature", "Humidity", "CO2", "Occupancy"]
        for i, name in enumerate(feature_names):
            if i < y_val.shape[1]:
                feature_rmse = np.sqrt(np.mean(np.square(y_val_original[:, i] - y_pred_original[:, i])))
                metrics[f"{name}_RMSE"] = feature_rmse
                
        return metrics

    def predict(self, data_sequence, steps=10):
        if not self._is_scaler_fitted():
            if len(self.training_data) >= 2:
                try:
                    feature_size = self.input_shape[1]  
                    valid_data = []
                    for item in self.training_data:
                        if isinstance(item, (list, np.ndarray)) and len(item) == feature_size:
                            valid_data.append(item)
                    if len(valid_data) >= 2:
                        self.scaler.fit(np.array(valid_data))
                        if not self._is_scaler_fitted():
                            return None
                    else:
                        return None
                except Exception:
                    return None
            else:
                return None

        if not isinstance(data_sequence, np.ndarray):
             data_sequence = np.array(data_sequence)

        expected_features = self.input_shape[1]
        if data_sequence.shape[1] != expected_features:
             return None

        timesteps = self.input_shape[0] 
        if len(data_sequence) < timesteps:
            padding_rows = timesteps - len(data_sequence)
            if len(data_sequence) > 0:
                 pad_values = np.mean(data_sequence, axis=0)
            else:
                 pad_values = np.zeros(expected_features)
            padding = np.tile(pad_values, (padding_rows, 1))
            data_sequence = np.vstack((padding, data_sequence))

        input_seq_np = data_sequence[-timesteps:]
        try:
            scaled_sequence = self.scaler.transform(input_seq_np)
        except Exception:
            return None

        current_sequence = scaled_sequence.reshape(1, timesteps, expected_features)
        if self.model is None:
            return None

        predictions_scaled = []
        try:
            for _ in range(steps):
                next_val_scaled = self.model.predict(current_sequence, verbose=0) 
                predictions_scaled.append(next_val_scaled[0]) 
                new_step_full_scaled = np.zeros((1, 1, expected_features))
                new_step_full_scaled[0, 0, :5] = next_val_scaled[0] 
                if expected_features > 5 and current_sequence.shape[2] == expected_features:
                     new_step_full_scaled[0, 0, 5:] = current_sequence[0, -1, 5:]
                current_sequence = np.append(current_sequence[:, 1:, :], new_step_full_scaled, axis=1)
        except Exception:
             return None

        predictions_scaled = np.array(predictions_scaled) 
        padded_predictions = np.zeros((steps, expected_features))
        padded_predictions[:, :5] = predictions_scaled

        if expected_features > 5:
             if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None and len(self.scaler.mean_) == expected_features:
                  padded_predictions[:, 5:] = self.scaler.mean_[5:] 
             elif len(self.training_data) > 0: 
                  last_vals = np.array(self.training_data[-1])
                  if len(last_vals) == expected_features:
                       padded_predictions[:, 5:] = last_vals[5:]

        try:
            final_predictions = self.scaler.inverse_transform(padded_predictions)
            for i in range(final_predictions.shape[0]):
                co2_val = final_predictions[i, 2]  
                occ_val = final_predictions[i, 3]  
                if expected_features > 5:  
                    final_predictions[i, 5] = calculate_kpiv(co2_val, occ_val)
            return final_predictions
        except Exception:
            return None

    def sync_with_cloud(self):
        try:
            cloud_model_url = f"https://your-cloud-storage.com/models/{self.classroom_id}_model.h5"
            cloud_model_info_url = f"https://your-cloud-storage.com/models/{self.classroom_id}_info.json"
            response = requests.get(cloud_model_info_url)
            if response.status_code == 200:
                cloud_info = response.json()
                cloud_update_time = cloud_info.get('last_update', 0)
                if cloud_update_time > self.last_update:
                    model_response = requests.get(cloud_model_url)
                    if model_response.status_code == 200:
                        with open(self.model_path, 'wb') as f:
                            f.write(model_response.content)
                        self.model = load_model(self.model_path)
                        self.last_update = cloud_update_time
                        return True
            return True
        except Exception:
            return False

class ThingSpeakManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.base_url = "https://api.thingspeak.com/update"
        self.last_upload_time = {}
        self.upload_interval = 15  

    def upload_data(self, classroom_id, data, model=None):
        current_time = time.time()
        if classroom_id in self.last_upload_time and \
           (current_time - self.last_upload_time[classroom_id]) < self.upload_interval:
            return False

        temperature = data[0]
        humidity = data[1]
        co2 = data[2]
        occupancy = data[3]  
        kpiv = data[5] if len(data) > 5 else 0  
        trend = data[6] if len(data) > 6 else 0  

        alert_status = 0
        if kpiv >= 1.0 or co2 > 1000:
            alert_status = 2  
        elif kpiv >= 0.8 or co2 > 800:
            alert_status = 1  

        api_key = self.api_keys.get(classroom_id)
        if not api_key:
            return False

        payload = {
            'api_key': api_key,
            'field1': temperature,
            'field2': humidity,
            'field3': co2,
            'field4': occupancy,
            'field5': kpiv,
            'field6': trend,
            'field7': alert_status,
        }
        
        if model and hasattr(model, 'performance_metrics') and model.performance_metrics:
            metrics = model.performance_metrics
            payload['field8'] = metrics.get("R_squared", 0.0)
        else:
            payload['field8'] = 0.0

        try:
            response = requests.get(self.base_url, params=payload)
            if response.status_code == 200:
                self.last_upload_time[classroom_id] = current_time
                return True
            return False
        except Exception:
            return False

    def upload_batch(self, classroom_data, models=None):
        results = {}
        for classroom_id, data_list in classroom_data.items():
            if data_list:
                latest_data = data_list[-1]
                model = None if models is None else models.get(classroom_id)
                results[classroom_id] = self.upload_data(classroom_id, latest_data, model)
                time.sleep(16)
        return results

class ClassroomMonitor:
    def __init__(self):
        try:
            self.ser_usb0 = serial.Serial(usb0_port, baud_rate, timeout=1)
            self.ser_usb1 = serial.Serial(usb1_port, baud_rate, timeout=1)
        except Exception as e:
            logging.error(f"Error opening serial ports: {e}")
            raise

        self.harmonizer = DataHarmonizer(window_size=DATA_HARMONIZATION_WINDOW)
        lstm_input_shape = (10, 8) 
        self.models = {
            'classroom1': LSTMModel('classroom1', input_shape=lstm_input_shape),
            'classroom2': LSTMModel('classroom2', input_shape=lstm_input_shape)
        }
        self.classroom_data = {
            'classroom1': [],
            'classroom2': []
        }
        self.last_cloud_upload = time.time()

        lcd.clear()
        lcd.write_string("Classroom")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Monitor v2.0")
        time.sleep(2)

        thingspeak_api_keys = {
            'classroom1': 'FUTC7Y9NKV47NROU',
            'classroom2': 'K8GEMD0ZWCWSU4SB'
        }
        self.thingspeak = ThingSpeakManager(thingspeak_api_keys)
        self.cloud_upload_interval = 120  

    def parse_data(self, data):
        try:
            temperature = int(data[data.index('a') + 1:data.index('b')])
            humidity = int(data[data.index('b') + 1:data.index('c')])
            co2 = int(data[data.index('c') + 1:data.index('d')])
            inside_count = int(data[data.index('d') + 1:data.index('e')])
            outside_count = int(data[data.index('e') + 1:data.index('f')])
            kpiv = 0.0
            if 'f' in data and 'g' in data:
                kpiv = float(data[data.index('f') + 1:data.index('g')])
            trend = 0.0
            if 'g' in data and 'h' in data:
                trend = float(data[data.index('g') + 1:data.index('h')])
            return [temperature, humidity, co2, inside_count, outside_count, kpiv, trend]
        except Exception:
            return None

    def collect_data(self, duration=60):
        start_time = time.time()
        classroom1_count = 0
        classroom2_count = 0
        lcd.clear()
        lcd.write_string("Collecting data")

        while time.time() - start_time < duration:
            if self.ser_usb0.in_waiting > 0:
                data = self.ser_usb0.readline().decode().strip()
                if data.startswith("a") and ('f' in data):
                    parsed = self.parse_data(data)
                    if parsed:
                        harmonized = self.harmonizer.harmonize('classroom1', parsed)
                        self.classroom_data['classroom1'].append(harmonized)
                        self.models['classroom1'].add_training_data(harmonized)
                        classroom1_count += 1

            if self.ser_usb1.in_waiting > 0:
                data = self.ser_usb1.readline().decode().strip()
                if data.startswith("a") and ('f' in data):
                    parsed = self.parse_data(data)
                    if parsed:
                        harmonized = self.harmonizer.harmonize('classroom2', parsed)
                        self.classroom_data['classroom2'].append(harmonized)
                        self.models['classroom2'].add_training_data(harmonized)
                        classroom2_count += 1

            if time.time() % 5 < 0.1:  
                lcd.clear()
                lcd.write_string(f"C1 Readings: {classroom1_count}")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"C2 Readings: {classroom2_count}")

            if time.time() - self.last_cloud_upload >= self.cloud_upload_interval:
                self.thingspeak.upload_batch(self.classroom_data, self.models)
                self.last_cloud_upload = time.time()

            time.sleep(0.1)

        return classroom1_count > 0 and classroom2_count > 0

    def send_model_updates(self, classroom_id):
        serial_conn = self.ser_usb0 if classroom_id == 'classroom1' else self.ser_usb1
        params = self.calculate_model_parameters(classroom_id)
        
        model = self.models.get(classroom_id)
        if model and hasattr(model, 'performance_metrics') and model.performance_metrics:
            metrics = model.performance_metrics
            params["r_squared"] = round(metrics.get("R_squared", 0), 2)
            params["rmse"] = round(metrics.get("RMSE", 0), 2)
            params["co2_accuracy"] = round(metrics.get("CO2_Critical_Threshold_Accuracy", 0), 2)
            params["kpiv_accuracy"] = round(metrics.get("KPIv_Critical_Threshold_Accuracy", 0), 2)

        command = "MODEL:"
        for key, value in params.items():
            command += f"{key}:{value},"
        command = command[:-1]

        try:
            serial_conn.write((command + "\n").encode())
            return True
        except Exception:
            return False

    def calculate_model_parameters(self, classroom_id):
        recent_data = self.classroom_data.get(classroom_id, [])
        if len(recent_data) < 10:
            return {
                "co2_weight": 0.6,
                "temp_weight": 0.3,
                "humidity_weight": 0.1,
                "trend_threshold": 0.7
            }

        recent_data = np.array(recent_data[-50:])
        kpiv_idx = 5  
        co2_corr = np.abs(np.corrcoef(recent_data[:, 2], recent_data[:, kpiv_idx])[0, 1])
        temp_corr = np.abs(np.corrcoef(recent_data[:, 0], recent_data[:, kpiv_idx])[0, 1])
        humidity_corr = np.abs(np.corrcoef(recent_data[:, 1], recent_data[:, kpiv_idx])[0, 1])

        co2_corr = 0.6 if np.isnan(co2_corr) else co2_corr
        temp_corr = 0.3 if np.isnan(temp_corr) else temp_corr
        humidity_corr = 0.1 if np.isnan(humidity_corr) else humidity_corr

        total = co2_corr + temp_corr + humidity_corr
        if total > 0:
            co2_weight = co2_corr / total
            temp_weight = temp_corr / total
            humidity_weight = humidity_corr / total
        else:
            co2_weight = 0.6
            temp_weight = 0.3
            humidity_weight = 0.1

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
        results = {}
        for classroom_id, model in self.models.items():
            if model.train():
                results[classroom_id] = True
                lcd.clear()
                lcd.write_string(f"{classroom_id}")
                lcd.cursor_pos = (1, 0)
                r_squared = model.performance_metrics.get("R_squared", 0)
                co2_acc = model.performance_metrics.get("CO2_Critical_Threshold_Accuracy", 0)
                lcd.write_string(f"R2:{r_squared:.2f} CO2:{co2_acc:.2f}")
                time.sleep(3)
                self.send_model_updates(classroom_id)
            else:
                results[classroom_id] = False
        return results

    def make_predictions(self, steps=10):
        predictions = {}
        for classroom_id, model in self.models.items():
            if len(self.classroom_data[classroom_id]) >= 10:
                data = np.array(self.classroom_data[classroom_id][-10:])
                pred = model.predict(data, steps=steps)
                predictions[classroom_id] = pred
        return predictions

    def score_predictions(self, predictions):
        scores = {}
        for classroom_id, pred in predictions.items():
            score = 0
            for p in pred:
                temp = p[0]
                hum = p[1] 
                co2 = p[2]
                occupancy = p[3]
                kpiv = p[5] if len(p) > 5 else calculate_kpiv(co2, occupancy)
                score += kpiv * 5
                score += abs(temp - 24) * 0.2
                score += (co2 / 1000) * 0.3
                score += occupancy * 0.1
            scores[classroom_id] = score
        return scores

    def get_recommendation(self):
        if not self.collect_data(duration=60):
            return "Not enough data"

        self.train_models()
        predictions = self.make_predictions(steps=10)
        if len(predictions) < 2:
            return "Not enough data for prediction"

        scores = self.score_predictions(predictions)
        if scores['classroom1'] < scores['classroom2']:
            recommendation = "Classroom 1"
            details = f"Score: {scores['classroom1']:.2f} vs {scores['classroom2']:.2f}"
        elif scores['classroom2'] < scores['classroom1']:
            recommendation = "Classroom 2"
            details = f"Score: {scores['classroom2']:.2f} vs {scores['classroom1']:.2f}"
        else:
            recommendation = "Both Equal"
            details = f"Score: {scores['classroom1']:.2f}"

        return recommendation, details

    def run(self):
        try:
            last_metrics_display = 0
            metrics_display_interval = 1800
            
            while True:
                current_time = time.time()
                if current_time - last_metrics_display >= metrics_display_interval:
                    for classroom_id, model in self.models.items():
                        if hasattr(model, 'performance_metrics') and model.performance_metrics:
                            lcd.clear()
                            lcd.write_string(f"{classroom_id} Metrics:")
                            lcd.cursor_pos = (1, 0)
                            r_squared = model.performance_metrics.get("R_squared", 0)
                            rmse = model.performance_metrics.get("RMSE", 0)
                            lcd.write_string(f"R2:{r_squared:.2f} RMSE:{rmse:.2f}")
                            time.sleep(2)
                            
                            lcd.clear()
                            lcd.write_string(f"{classroom_id} Accuracy:")
                            lcd.cursor_pos = (1, 0)
                            co2_acc = model.performance_metrics.get("CO2_Critical_Threshold_Accuracy", 0)
                            kpiv_acc = model.performance_metrics.get("KPIv_Critical_Threshold_Accuracy", 0)
                            lcd.write_string(f"CO2:{co2_acc:.2f} KPIv:{kpiv_acc:.2f}")
                            time.sleep(2)
                    
                    last_metrics_display = current_time

                result = self.get_recommendation()
                if isinstance(result, tuple):
                    recommendation, details = result
                else:
                    recommendation = result
                    details = ""

                lcd.clear()
                lcd.write_string("Better Room:")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(recommendation)

                if time.time() - self.last_cloud_upload >= self.cloud_upload_interval:
                    self.thingspeak.upload_batch(self.classroom_data, self.models)
                    self.last_cloud_upload = time.time()

                time.sleep(300)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            lcd.clear()
            GPIO.cleanup()

if __name__ == "__main__":
    monitor = ClassroomMonitor()
    monitor.run()