import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

def create_model(input_shape):
    model = Sequential()
    
    # LSTM layers for sequential data
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.4))
    
    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # 3 outputs: accept, reject (2 persons), reject (time shift)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# تنظیمات عمومی
sampling_rate = 100  # نرخ نمونه‌برداری (در هر ثانیه)
duration = 10  # مدت زمان سیگنال (ثانیه)
t = np.linspace(0, duration, sampling_rate * duration)  # محور زمان

# تابع برای تولید سیگنال مصنوعی ECG ساده
def generate_ecg_signal(frequency=1.0, noise_level=0.1):
    signal = np.sin(2 * np.pi * frequency * t)  # موج سینوسی به عنوان سیگنال ECG مصنوعی
    noise = noise_level * np.random.normal(size=t.shape)  # نویز تصادفی
    return signal + noise

# تولید داده‌ها برای هر حالت
def generate_data():
    # حالت 1: یک شخص (پذیرش)
    signal_person1 = generate_ecg_signal(frequency=1.0)
    
    # حالت 2: دو شخص مختلف (رد)
    signal_person2 = generate_ecg_signal(frequency=1.5)
    combined_signal = (signal_person1 + signal_person2) / 2
    
    # حالت 3: یک شخص با شیفت زمانی (رد)
    time_shift = int(0.2 * sampling_rate)  # شیفت زمانی (200 میلی‌ثانیه)
    shifted_signal = np.roll(signal_person1, shift=time_shift)
    
    # برچسب‌ها: 0 = پذیرش, 1 = رد (دو شخص), 2 = رد (شیفت زمانی)
    data = [
        (signal_person1, 0),  # پذیرش
        (combined_signal, 1),  # رد (دو شخص)
        (shifted_signal, 2)  # رد (شیفت زمانی)
    ]
    
    return data

# نمایش داده‌های مصنوعی
def plot_signals(data):
    plt.figure(figsize=(12, 8))
    for i, (signal, label) in enumerate(data):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, signal)
        plt.title(f'Signal {i+1} - Label: {label}')
    plt.tight_layout()
    plt.show()

# تولید و نمایش داده‌ها
data = generate_data()
plot_signals(data)

