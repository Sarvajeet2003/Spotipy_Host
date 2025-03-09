from collections import deque
import pyttsx3
import cv2
import time
import numpy as np
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
import logging
import os
import json  # <--- NEW import for writing JSON

# Setup logging
logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Spotify API credentials

SPOTIFY_CLIENT_ID = '5570b7ff4a454259b4b8ac9ca0ef90f9'
SPOTIFY_CLIENT_SECRET = '297b1a50c4db45a287ba5bcdc0f684af'
SPOTIFY_REDIRECT_URI = 'http://localhost:8888/callback/'

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state"
))

# Map emotions to genres
emotion_genres = {
    'Angry': 'rock',
    'Happy': 'pop',
    'Sad': 'acoustic',
    'Surprise': 'indie',
    'Neutral': 'ambient',
    'Fear': 'classical',
    'Disgust': 'blues'
}

# Track played songs for each emotion
played_songs = {emotion: set() for emotion in emotion_genres.keys()}

# Load your trained model
model_path = "Model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
your_model = load_model(model_path)

# Emotion labels and colors
emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Fear': (255, 0, 0),
    'Happy': (255, 255, 0),
    'Sad': (0, 255, 255),
    'Surprise': (255, 0, 255),
    'Neutral': (128, 128, 128)
}

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Variables for summary and smooth predictions
start_time = time.time()
emotion_summary = []
emotion_window = deque(maxlen=30)  # Store last 30 predictions
current_song_start_time = None
song_duration = 30  # Minimum duration for a song to play in seconds
frame_count = 0
frame_skip_rate = 5  # Process every 5th frame

while True:
    try:
        ret, frame = cap.read()
        if not ret or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        frame_count += 1
        if frame_count % frame_skip_rate != 0:
            continue

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face ROI and preprocess
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 3)

            # Predict emotion
            preds = your_model.predict(reshaped_face)
            emotion_idx = preds.argmax()
            confidence = preds[0][emotion_idx]

            if confidence < 0.6:
                emotion = "Uncertain"
            else:
                emotion = emotion_labels[emotion_idx]

            # Append to summaries and sliding window
            emotion_summary.append(emotion)
            emotion_window.append(emotion)
            smooth_emotion = max(set(emotion_window), key=emotion_window.count)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          emotion_colors.get(emotion, (255, 255, 255)), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, emotion_colors.get(emotion, (255, 255, 255)), 2)

        # Display the frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Check if 5 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 20:
            if emotion_summary:
                # Analyze emotions and provide summary
                emotion_summary_counts = {
                    label: emotion_summary.count(label) for label in set(emotion_summary)
                }
                dominant_emotion = max(emotion_summary_counts, key=emotion_summary_counts.get)
                print("Emotion Summary (last 5 seconds):")
                for label, count in emotion_summary_counts.items():
                    print(f"{label}: {count}")

                print(f"Dominant Emotion: {dominant_emotion}")

                # Text-to-speech feedback
                engine = pyttsx3.init()
                engine.say(f"Dominant Emotion: {dominant_emotion}")
                engine.runAndWait()

                # Play songs from user account for the detected emotion
                current_time = time.time()
                if not current_song_start_time or (current_time - current_song_start_time) >= song_duration:
                    current_song_start_time = current_time

                    if dominant_emotion in emotion_genres:
                        genre = emotion_genres[dominant_emotion]
                        results = sp.search(q=f'genre:{genre}', type='track', limit=20)
                        if results['tracks']['items']:
                            available_tracks = [track['uri'] for track in results['tracks']['items']]

                            # Remove already played tracks for the emotion
                            unplayed_tracks = list(set(available_tracks) - played_songs[dominant_emotion])

                            if not unplayed_tracks:
                                print(f"All songs for {genre} have been played, resetting history.")
                                played_songs[dominant_emotion].clear()
                                unplayed_tracks = available_tracks

                            # Randomly select a song
                            selected_track = random.choice(unplayed_tracks)
                            played_songs[dominant_emotion].add(selected_track)

                            # Get track name & album art
                            try:
                                track_info = sp.track(selected_track)
                                track_name = track_info.get('name', 'Unknown Track')

                                # album.images is usually a list of dicts with different sizes
                                album_images = track_info['album']['images']
                                if album_images:
                                    album_art = album_images[0]['url']  # Largest image
                                else:
                                    album_art = ''
                            except Exception as e:
                                print("Error retrieving track info:", e)
                                track_name = 'Unknown Track'
                                album_art = ''

                            # Write the info to a JSON file so Flask can read it
                            with open('current_track.json', 'w', encoding='utf-8') as f:
                                json.dump({
                                    "track_name": track_name,
                                    "album_art": album_art
                                }, f, ensure_ascii=False)

                            # Play the selected song
                            devices = sp.devices()
                            if devices['devices']:
                                device_id = devices['devices'][0]['id']
                                sp.start_playback(device_id=device_id, uris=[selected_track])
                                print(f"Playing a random {genre} song for {dominant_emotion}: {track_name}")
                            else:
                                print("No Spotify devices found.")
                        else:
                            print(f"No songs found for genre: {genre}")

            else:
                print("No faces detected in the last 5 seconds.")

            # Reset variables for the next window
            start_time = time.time()
            emotion_summary = []

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        logging.error("Error occurred", exc_info=True)
        break

cap.release()
cv2.destroyAllWindows()