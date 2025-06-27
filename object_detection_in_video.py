import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


try:
    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    print(" Model loaded successfully!")
except Exception as e:
    print(" Model loading failed!")
    print("Error:", e)
    exit()


video_path = (r"C:\Users\nnand\Downloads\Car street.mp4")  
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = model(input_tensor)
    detections = {k: v.numpy() for k, v in detections.items()}


    boxes = detections['detection_boxes'][0]
    classes = detections['detection_classes'][0].astype(np.int32)
    scores = detections['detection_scores'][0]

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 3:  # Class 3 = car
            ymin, xmin, ymax, xmax = boxes[i]
            left, right = int(xmin * width), int(xmax * width)
            top, bottom = int(ymin * height), int(ymax * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
