import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from improvedModel import ImprovedCustomCNN 
import matplotlib.pyplot as plt

# Load the model
def load_model(model_path):
    num_classes = 7  
    model = ImprovedCustomCNN(num_classes=num_classes, activation='ReLU')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  
    return image

# Predict the emotion
def predict_emotion(model, image_tensor, original_image):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 
        
        # Display results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.convert('RGB'))  # Convert grayscale to RGB for display
        plt.title("Tested Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(emotions))
        plt.barh(y_pos, probabilities, align='center', alpha=0.5)
        plt.yticks(y_pos, emotions)
        plt.xlabel('Confidence (%)')
        plt.title('Emotion Predictions')
        
        plt.tight_layout()
        plt.show()

        print(f"Predicted emotion: {emotions[predicted.item()]} with {probabilities[predicted.item()].item():.2f}% confidence")
        print("Confidence levels for all emotions:")
        for i, emotion in enumerate(emotions):
            print(f"{emotion}: {probabilities[i].item():.2f}%")

if __name__ == "__main__":
    model_path = 'C://Users//andre//Documents//emotion_recognition_model.pth'  
    image_path = 'C://Users//andre//Documents//neutral1.jpg'  
    
    model = load_model(model_path)
    original_image = Image.open(image_path) 
    image_tensor = preprocess_image(image_path)
    predict_emotion(model, image_tensor, original_image)