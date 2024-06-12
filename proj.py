import streamlit as st
from streamlit_webrtc import webrtc_streamer

import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import requests
import json
import webbrowser
import base64

# Spotify API credentials
client_id = '6e5f7e8ca2514f26b7b5e6b1d99d77c8'
client_secret = 'fcadc302e4094f4482b9de2d07c2b29b'

# Get Spotify OAuth token
def get_spotify_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_headers = {
        'Authorization': 'Basic ' + base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(auth_url, headers=auth_headers, data=auth_data)
    return response.json().get('access_token')

def get_recommendations(emotion, artist, token):
    search_url = f"https://api.spotify.com/v1/search?q={emotion}%20{artist}&type=track&limit=10"
    search_headers = {
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(search_url, headers=search_headers)
    return response.json()

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")

with col2:
    st.write("")

with col3:
    st.write("")

st.title("Moosic")
st.write('Moosic is an emotion detection-based music recommendation system. To get recommended songs, start by allowing mic and camera for this web app.')

model = load_model("C:/Users/deves/OneDrive/Desktop/project/model.h5")
label = np.load("C:/Users/deves/OneDrive/Desktop/project/labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

if "run" not in st.session_state:
    st.session_state["run"] = "true"
try:
    detected_emotion = np.load("C:/Users/deves/OneDrive/Desktop/project/detected_emotion.npy")[0]
except:
    detected_emotion = ""

if not detected_emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)  # Flipping the frame from left to right

        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        lst = []

        # Storing Landmark data
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        np.save("C:/Users/deves/OneDrive/Desktop/project/detected_emotion.npy", np.array([pred]))

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Enter your preferred language")
artist = st.text_input("Enter your preferred artist")

if lang and artist and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionDetector)

btn = st.button("Recommend music")

if btn:
    if not detected_emotion:
        st.warning("Please let me capture your emotion first!")
        st.session_state["run"] = "true"
    else:
        
        if detected_emotion == "sad":
            emotion = "happy"
        elif detected_emotion == "angry":
            emotion = "calm"
        else:
            emotion = detected_emotion

        token = get_spotify_token(client_id, client_secret)
        recommendations = get_recommendations(emotion, artist, token)

        if "tracks" in recommendations:
            st.write("Recommended Top 10 " + emotion + " songs:")
            for track in recommendations["tracks"]["items"]:
                st.write(track["name"] + " by " + ", ".join([artist["name"] for artist in track["artists"]]))
                st.audio(track["preview_url"])

                track_url = track["external_urls"]["spotify"]
                st.markdown(f"[Listen full song on Spotify Web Player]({track_url})")

        np.save("C:/Users/deves/OneDrive/Desktop/project/detected_emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

# Streamlit Customization
st.markdown(""" <style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
