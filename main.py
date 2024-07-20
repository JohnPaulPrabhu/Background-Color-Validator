import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_video_path = 'output_video.mp4'


class BlurClassifier(nn.Module):
    def __init__(self, dropout_rate):
        super(BlurClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

# Load your pre-trained model
model = BlurClassifier(dropout_rate=0.2).to(device)
model.load_state_dict(torch.load('final_best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transform to preprocess the frame
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assume you have a function is_blurry that takes an image and returns True if it is blurry, otherwise False
def is_blurry(frame):
    # Implement your model inference here
    frame = Image.fromarray(frame)
    frame = transform(frame).unsqueeze(0)
    with torch.no_grad():
        pred = model(frame.to(device))
        _, predicted = torch.max(pred, 1)
        print(predicted)
    return predicted


# Initialize video capture from the default camera
cap = cv2.VideoCapture("test3.mp4")

# Get the properties of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Check if the frame is blurry
    if not is_blurry(frame):
        label = "Blurry"
        color = (0, 0, 255)  # Red color for blurry frames
    else:
        label = "Not Blurry"
        color = (0, 255, 0)  # Green color for not blurry frames

    # Draw the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Write the processed frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Blur Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
