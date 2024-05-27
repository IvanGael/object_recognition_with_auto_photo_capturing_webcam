import cv2
import numpy as np
import torch
from PIL import Image

def capture_photo():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Countdown timer
    countdown = 20
    while countdown > 0:
        ret, frame = cap.read()
        cv2.putText(frame, str(countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('Capture Photo', frame)
        cv2.waitKey(1000)  # Wait for 1 second
        countdown -= 1
    
    # Capture photo
    ret, frame = cap.read()
    cv2.imwrite('captured_photo.jpg', frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Recognize object in captured photo
    recognize_object()

def recognize_object():
    # Load captured photo
    image_path = 'captured_photo.jpg'
    image = Image.open(image_path)

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform object detection
    results = model(image)

    # Draw bounding boxes and labels on the image
    results.render()
    rendered_image = np.array(results.ims[0])

    # Display the image with bounding boxes and labels
    cv2.imshow('Recognized Object', rendered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Capture photo using webcam
    capture_photo()

if __name__ == "__main__":
    main()
