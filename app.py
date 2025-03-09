from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
import cv2
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
import os
import json
from collections import deque
import time
import threading

app = Flask(__name__)

# Spotify configuration with environment variables
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '5570b7ff4a454259b4b8ac9ca0ef90f9')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '297b1a50c4db45a287ba5bcdc0f684af')
SPOTIFY_REDIRECT_URI = os.environ.get('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback/')

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state"
))

# Emotion configuration
emotion_genres = {
    'Angry': 'rock',
    'Happy': 'pop',
    'Sad': 'acoustic',
    'Surprise': 'indie',
    'Neutral': 'ambient',
    'Fear': 'classical',
    'Disgust': 'blues'
}

played_songs = {emotion: set() for emotion in emotion_genres.keys()}
try:
    model = load_model("Model.h5")  # Ensure Model.h5 exists in your project
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model for testing deployment
    model = None
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
    face_cascade = None

# Shared variables with thread safety
emotion_summary = []
emotion_window = deque(maxlen=30)
current_song_start_time = time.time()  # Initialize with current time
current_song_duration_ms = 0  # Track the full duration of current song
current_song_uri = None  # Track the current song URI
next_emotion_to_play = None  # Store the next emotion to play when current song ends
paused = False
pause_lock = threading.Lock()
song_duration = 60  # Change song every 60 seconds (1 minute)

# Background thread for periodic song changes
def song_change_thread():
    global current_song_start_time, current_song_duration_ms, current_song_uri, next_emotion_to_play
    while True:
        try:
            with pause_lock:
                if not paused:
                    # Check if 1 minute has passed since the last song change
                    elapsed = time.time() - current_song_start_time
                    if elapsed >= song_duration:
                        # If we have a next emotion queued up, play it
                        if next_emotion_to_play and next_emotion_to_play in emotion_genres:
                            play_song_for_emotion(next_emotion_to_play)
                            next_emotion_to_play = None
                        # Otherwise, check if we should play based on current emotion
                        elif emotion_window:
                            # Get the most frequent emotion
                            emotions_list = list(emotion_window)
                            emotion_counts = {}
                            for e in emotions_list:
                                if e in emotion_genres:
                                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                            
                            if emotion_counts:
                                dominant = max(emotion_counts, key=emotion_counts.get)
                                play_song_for_emotion(dominant)
                    else:
                        # Update current song information if needed
                        try:
                            playback = sp.current_playback()
                            if playback and playback.get('item') and playback['item'].get('uri') != current_song_uri:
                                current_song_uri = playback['item']['uri']
                                print(f"Now playing: {playback['item']['name']}")
                        except Exception as e:
                            print(f"Error updating song info: {e}")
        except Exception as e:
            print(f"Error in song change thread: {e}")
        time.sleep(2)  # Check every 2 seconds

def play_song_for_emotion(emotion):
    """Play a song for the given emotion"""
    global current_song_start_time, current_song_uri
    
    if emotion not in emotion_genres:
        return False
        
    genre = emotion_genres[emotion]
    try:
        # Check for active devices first
        devices = sp.devices()
        if not devices['devices']:
            print("No active Spotify devices found. Please open Spotify app.")
            return False
            
        device_id = devices['devices'][0]['id']
        
        # Search for tracks with the genre
        results = sp.search(q=f'genre:{genre}', type='track', limit=20)
        if results['tracks']['items']:
            available_tracks = [t['uri'] for t in results['tracks']['items']]
            unplayed = list(set(available_tracks) - played_songs[emotion])
            
            if not unplayed:
                played_songs[emotion].clear()
                unplayed = available_tracks

            selected = random.choice(unplayed)
            played_songs[emotion].add(selected)
            
            # Start playback with selected track
            sp.start_playback(device_id=device_id, uris=[selected])
            current_song_uri = selected
            current_song_start_time = time.time()
            print(f"Playing {genre} song for {emotion} emotion: {selected}")
            return True
        else:
            print(f"No tracks found for genre: {genre}")
    except spotipy.exceptions.SpotifyException as se:
        print(f"Spotify API error: {se}")
    except Exception as e:
        print(f"Error playing song: {e}")
    
    return False

# Start background thread
song_thread = threading.Thread(target=song_change_thread, daemon=True)
song_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global next_emotion_to_play
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'dominant_emotion': 'Happy',  # Default emotion
            'emotion_counts': {e: 1 for e in emotion_labels}
        })
    
    try:
        # Decode frame from client
        frame_data = request.json['frame'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with pause_lock:
            if paused:
                return jsonify({
                    'dominant_emotion': max(set(emotion_window), key=emotion_window.count) if emotion_window else "No Face",
                    'emotion_counts': {e: emotion_summary.count(e) for e in emotion_labels if e in emotion_summary}
                })

        # Process frame for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({
                'dominant_emotion': max(set(emotion_window), key=emotion_window.count) if emotion_window else "No Face",
                'emotion_counts': {e: emotion_summary.count(e) for e in emotion_labels if e in emotion_summary}
            })

        detected_emotions = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            # Convert BGR to RGB (OpenCV uses BGR by default)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize and normalize
            resized_face = cv2.resize(face_rgb, (48, 48)) / 255.0

            # Reshape for model input (1, 48, 48, 3)
            reshaped_face = resized_face.reshape(1, 48, 48, 3)
            
            preds = model.predict(reshaped_face)
            emotion = emotion_labels[preds.argmax()] if preds.max() >= 0.6 else "Uncertain"
            detected_emotions.append(emotion)
        
        # Only process valid emotions (those in our mapping)
        valid_emotions = [e for e in detected_emotions if e in emotion_genres]
        
        if valid_emotions:
            with pause_lock:
                for emotion in valid_emotions:
                    emotion_summary.append(emotion)
                    emotion_window.append(emotion)
                    
                    # Get current dominant emotion
                    emotions_list = list(emotion_window)
                    emotion_counts = {}
                    for e in emotions_list:
                        if e in emotion_genres:
                            emotion_counts[e] = emotion_counts.get(e, 0) + 1
                    
                    if emotion_counts:
                        current_dominant = max(emotion_counts, key=emotion_counts.get)
                        
                        # Check if dominant emotion has changed significantly
                        if len(emotion_window) >= 10:
                            dominant_count = emotion_counts.get(current_dominant, 0)
                            total_valid = sum(emotion_counts.values())
                            
                            # If dominant emotion represents at least 60% of recent emotions,
                            # queue it up for the next song change
                            if dominant_count / total_valid >= 0.6:
                                next_emotion_to_play = current_dominant
                                print(f"Emotion changed to {current_dominant}, will play after current song ends")

        dominant_emotion = max(set(emotion_window), key=emotion_window.count) if emotion_window else "Uncertain"
        
        return jsonify({
            'dominant_emotion': dominant_emotion,
            'emotion_counts': {e: emotion_summary.count(e) for e in emotion_labels if e in emotion_summary}
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global paused
    with pause_lock:
        paused = not paused
    return jsonify({'paused': paused})

@app.route('/current_track')
def current_track():
    try:
        playback = sp.current_playback()
        if playback and playback.get('item'):
            return jsonify({
                'track_name': playback['item']['name'],
                'album_art': playback['item']['album']['images'][0]['url'] if playback['item']['album']['images'] else ''
            })
    except Exception as e:
        print(f"Error getting current track: {e}")
    return jsonify({'track_name': 'No track playing', 'album_art': ''})

@app.route('/playback_state')
def playback_state():
    try:
        playback = sp.current_playback()
        if playback and playback.get('item'):
            return jsonify({
                'progress_ms': playback['progress_ms'],
                'duration_ms': playback['item']['duration_ms'],
                'is_playing': playback['is_playing']
            })
    except Exception as e:
        print(f"Error getting playback state: {e}")
    return jsonify({'progress_ms': 0, 'duration_ms': 0, 'is_playing': False})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, threaded=True)