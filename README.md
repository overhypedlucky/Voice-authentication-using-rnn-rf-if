🎙️ Ensemble Voice Authentication System
📌 Description

This project is based on a voice authentication system that verifies a user using their voice. The main idea behind this project is to improve security by using multiple models instead of relying on a single method. In this system, three different techniques are used: RNN (LSTM), Random Forest, and Isolation Forest.

First, the system takes an audio input from the user and processes it. Important features like MFCC and Mel Spectrogram are extracted from the voice signal. These features help in understanding the unique characteristics of a person’s voice.

After feature extraction, the data is passed through three models. The RNN model is used to understand the sequence and pattern of speech over time. The Random Forest model helps in classification based on extracted features. The Isolation Forest model is used to detect unusual or unknown voice inputs, which adds an extra layer of security.

The final result is calculated by combining the outputs of all three models. This combined decision makes the system more accurate and reliable. If the final score is above a certain value, the user is authenticated; otherwise, access is denied.

🚀 Key Features
Uses multiple models for better accuracy
Provides secure voice-based authentication
Detects unknown or suspicious inputs
Simple and effective system design


🎯 Where it can be used
This system can be used in banking, mobile security, login systems, and other applications where user authentication is required.
👨‍💻 Author
Lucky
