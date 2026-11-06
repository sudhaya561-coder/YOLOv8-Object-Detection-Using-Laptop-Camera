# WORKSHOP 2 : YOLOv8-Object-Detection-Using-Laptop-Camera

### NAME : UDHAYA S
### REG NO : 212225230287

# Aim
To access the laptop camera, capture an image, and detect objects using YOLOv8.

# Requirements
Anaconda
Jupyter Notebook
Laptop Camera

# Steps
Open Jupyter Notebook using Anaconda.
Access your laptop camera using OpenCV.
Wait for 5 seconds and capture an image.
Display the captured image.
Apply YOLOv8 object detection.
Display the detected image with bounding boxes and labels.

# Algorithm

Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects

# Student Task
Capture your own image using the laptop camera.
Detect the objects present in the image.
Display the final output.
Perform the experiment with 3 different scenes.

# Submission
Jupyter Notebook (.ipynb)
Original captured images
YOLOv8 output images
Screenshot of the final result

# PROGRAM :

~~~python

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load a more accurate YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

else:
    video_writer = None

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model(
                frame,
                conf=0.60,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # Initialize video writer
            if video_writer is None:
                h, w = annotated_frame.shape[:2]

                video_writer = cv2.VideoWriter(
                    "YOLO_Detection_Output.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (w, h)
                )

            # Save detected frame
            video_writer.write(annotated_frame)

            # Display output in Jupyter
            rgb_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            clear_output(wait=True)

            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_frame)
            plt.title("YOLO Real-Time Object Detection")
            plt.axis("off")
            plt.show()

    except KeyboardInterrupt:
        print("Detection stopped.")

    finally:
        # Release webcam
        cap.release()

        # Release video writer
        if video_writer is not None:
            video_writer.release()

        print("Webcam released.")
        print("Video saved as YOLO_Detection_Output.mp4")
~~~
# OUTPUT :

<img width="798" height="677" alt="image" src="https://github.com/user-attachments/assets/eb687a00-86c0-49c2-ac05-35fe5403cf05" />
