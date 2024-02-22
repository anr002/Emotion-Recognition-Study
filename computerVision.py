import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from improvedModel import ImprovedCustomCNN
from pytube import YouTube
import os
import imageio
from collections import deque

def load_model(model_path):
    model = ImprovedCustomCNN(num_classes=7, activation='ReLU')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def download_youtube_video(youtube_url, save_path='downloaded_video.mp4'):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(filename=save_path)
    return save_path

def preprocess_image(image):
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((48, 48)),
        ToTensor(),
    ])
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image

def predict_emotion(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        return emotions[predicted.item()], probabilities[predicted.item()].item()

emotion_history = deque(maxlen=10)  # Adjust maxlen to change the smoothing window size

def update_emotion_history(emotion, confidence):
    global emotion_history
    emotion_history.append((emotion, confidence))

def get_smoothed_emotion():
    if not emotion_history:
        return None, None  # In case there are no emotions in the history yet
    
    # Calculate the most frequent emotion in the history
    emotions, confidences = zip(*emotion_history)
    avg_confidence = np.mean(confidences)
    most_common_emotion = max(set(emotions), key=emotions.count)
    
    return most_common_emotion, avg_confidence

def process_video_stream(model_path, video_path, start_time=0, end_time=None):
    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(video_path)
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the video frame rate
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out_video = cv2.VideoWriter('Results/processed_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    # Create a GIF writer object
    gif_writer = imageio.get_writer('Results/processed_video.gif', fps=fps)
    
    while True:
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        if end_time is not None and current_time >= end_time * 1000:
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=7, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = preprocess_image(face_img)
            emotion, confidence = predict_emotion(model, face_img)
            
            # Update the emotion history with the new prediction
            update_emotion_history(emotion, confidence)
            
            # Get the smoothed emotion
            smoothed_emotion, smoothed_confidence = get_smoothed_emotion()
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            # Use the smoothed_emotion and smoothed_confidence for display
            cv2.putText(frame, f"{smoothed_emotion}: {smoothed_confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Convert the frame from BGR to RGB (imageio uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_writer.append_data(frame_rgb)
        
        # Write the processed frame to the MP4 file
        out_video.write(frame)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out_video.release()  # Finalize the video file
    cv2.destroyAllWindows()
    gif_writer.close()  # Close the writer to finalize the GIF
    os.remove(video_path)
    print(f"Deleted video file: {video_path}")
    print("GIF saved in 'Results/processed_video.gif'")
    print("Video saved in 'Results/processed_video.mp4'")


if __name__ == "__main__":
    model_path = 'C://Users//andre//Documents//emotion_recognition_model.pth'
    youtube_url = 'https://www.youtube.com/watch?v=GjTAOOrvGd0&pp=ygUfY29uYW4gb2JyaWVuIHRyYXZlbGluZyBmdW5uaWVzdA%3D%3D'
    start_time = 12  # Start time in seconds
    end_time = 27  # End time in seconds, adjust as needed

    save_path = 'G:\\Documents\\downloaded_video.mp4'  # Adjusted save path
    video_path = download_youtube_video(youtube_url, save_path=save_path)
    process_video_stream(model_path, video_path, start_time, end_time)