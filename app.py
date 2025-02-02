from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
import pygame

app = Flask(__name__)
app.secret_key = "secret_key"  # Add a secret key for flash messages
# Paths
weight_path = "yolov3_training_2000.weights"
config_path = "yolov3_testing.cfg"
sound_path = "s.mp3"
# Initialize Pygame mixer
pygame.mixer.init()
# Load Yolo
net = cv2.dnn.readNet(weight_path, config_path)
# Name custom object
classes = ["Weapon"]
# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load and play sound
def play_sound():
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

# Stop sound
def stop_sound():
    pygame.mixer.music.stop()

# Function to resize frame
def resize_frame(frame, new_width=None, new_height=None):
    if new_width is None and new_height is None:
        return frame

    if new_width is None:
        aspect_ratio = new_height / float(frame.shape[0])
        new_width = int(frame.shape[1] * aspect_ratio)
    elif new_height is None:
        aspect_ratio = new_width / float(frame.shape[1])
        new_height = int(frame.shape[0] * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame

# Function to process frame
def process_frame(img, frame_count):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    weapon_detected = False
    detected_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
            weapon_detected = True
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label == "gun" else (0, 0, 255)  # Green for gun, red for others
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), font, 1, color, 1)
            detected_boxes.append([x, y, w, h, confidence])

    if weapon_detected:
        cv2.putText(img, f"Weapon detected - Frame: {frame_count}", (10, height - 30), font, 1.5, (0, 255, 0), 2)
        play_sound()
    else:
        cv2.putText(img, "No weapon detected", (10, height - 80), font, 3, (0, 0, 255), 3)  # Larger font and position adjusted

    return img, weapon_detected, detected_boxes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded file to a temporary location
            file_path = "static/uploads/" + file.filename
            file.save(file_path)
            return redirect(url_for('process_file', filename=file.filename))
    return render_template('index.html')

@app.route('/process_file/<filename>')
def process_file(filename):
    # Process the uploaded file
    input_path = "static/uploads/" + filename
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_frame, detected, _ = process_frame(frame, frame_count)
            resized_frame = resize_frame(processed_frame, new_width=640, new_height=480)  # Adjust the new_width and new_height as needed

            cv2.imshow("Frame", resized_frame)
            cv2.waitKey(1000)  # Adjust the delay time as needed (1000 ms = 1 second)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return 'Processing complete'
    else:
        return 'Invalid input path'

if __name__ == "__main__":
    app.run(debug=True)

