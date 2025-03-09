# Spotipy - Emotion-Based Music Player

Spotipy is a web application that detects user emotions using a machine learning model and plays relevant songs based on the detected emotion.

## Features
- **Emotion Detection**: Uses a deep learning model (`Model.h5`) to analyze emotions.
- **Flask Backend**: Manages API requests and serves the web application.
- **Web Interface**: Provides an interactive UI for users to input data and receive song recommendations.
- **Spotify Integration (Upcoming)**: Plans to connect with Spotify API to play songs based on detected emotions.

## Project Structure
```
Spotipy-main/
│── Model.h5              # Pre-trained deep learning model for emotion detection
│── app.py                # Main Flask application
│── emotion_detection.py  # Emotion detection script
│── requirements.txt      # Dependencies required for the project
│── runtime.txt           # Runtime environment specifications
│── static/
│   └── image.png         # Static assets
│── templates/
│   └── index.html        # Frontend UI
```

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/sarvajeet2003/Spotipy.git
cd Spotipy-main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
The application will start on `http://127.0.0.1:5000/`

## Dependencies
Ensure you have the following installed:
- Python 3.7+
- Flask
- TensorFlow/Keras
- OpenCV

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- **Integrate Spotify API** to fetch and play songs.
- **Improve Emotion Recognition Accuracy** with a better model.
- **Enhance UI** with React-based frontend.

## Contributors
- **Sarvajeeth U K** - Developer

## License
This project is free to use by all
