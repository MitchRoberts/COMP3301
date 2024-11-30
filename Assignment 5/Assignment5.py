import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from PIL import Image


class FaceTracking:
    def __init__(self, mode="face"):
        # Load pre-trained Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_nose.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

        # Check if Haar cascades are loaded properly
        if self.face_cascade.empty() or self.eye_cascade.empty() or self.nose_cascade.empty() or self.mouth_cascade.empty():
            raise ValueError("Failed to load Haar cascades. Ensure the files exist in the correct directory.")

        # Set mode: "face" or "corners"
        self.mode = mode
        self.input_type = None

    def load_image_via_dialog(self):
        """
        Dynamically select an image via a file explorer to create histograms for
        """
        app = QApplication(sys.argv)
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)

        if not file_path:
            raise ValueError("No image file selected!")

        return self.load_image(file_path)

    def load_image(self, path):
        """
        Load image from path and return as a numpy array
        """
        img = Image.open(path)
        if img.mode == 'L':
            return np.array(img) / 255.0
        else:
            return np.array(img.convert('RGB')) / 255.0

    def detect_corners(self, frame):
        """
        Detect corners on the input frame and highlight them using the Harris Corner Detection method.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # Harris Corner Detection
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate the result to enhance visualization
        dst = cv2.dilate(dst, None)

        # Apply a threshold to identify strong corners
        threshold = 0.01 * dst.max()
        corners = (dst > threshold)

        # Highlight the corners on the original frame
        frame[corners] = [0, 0, 255]  # Red color for Harris corners

        return frame



    def detect_faces(self, frame):
        """
        Detect faces and facial features and draw bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            
            #Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #Define regions of interest (ROI) for additional features
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            #Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            #Detect nose
            nose = self.nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (nx, ny, nw, nh) in nose:
                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)

            #Detect mouth
            mouth = self.mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (mx, my, mw, mh) in mouth:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 255), 2)

        return frame

    def start(self):
        if self.input_type == "image":
            # Load a static image for processing
            try:
                static_image = self.load_image_via_dialog()
                static_image = (static_image * 255).astype(np.uint8)  # Convert back to uint8 for OpenCV
                static_image = cv2.cvtColor(static_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

                if self.mode == "corners":
                    static_image = self.detect_corners(static_image)
                elif self.mode == "face":
                    static_image = self.detect_faces(static_image)

                # Create a resizable OpenCV window
                cv2.namedWindow("Image Processing", cv2.WINDOW_NORMAL)
                cv2.imshow("Image Processing", static_image)
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                cv2.destroyAllWindows()
            except ValueError as e:
                print(e)
            return

        # Default to webcam for processing
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Failed to open webcam. Check if the webcam is connected or in use by another application.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Perform operation based on the current mode
            if self.mode == "face":
                frame = self.detect_faces(frame)
            elif self.mode == "corners":
                frame = self.detect_corners(frame)

            # Display the video frame
            cv2.imshow("Webcam Processing", frame)

            # Exit if 'q' is pressed or window is closed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if cv2.getWindowProperty("Webcam Processing", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ask user for input type
    input_type = input("Enter input type (webcam or image): ").strip().lower()
    if input_type not in ["webcam", "image"]:
        print("Invalid input type. Defaulting to webcam.")
        input_type = "webcam"

    # Ask user for mode
    mode = input("Enter mode (face or corners): ").strip().lower()
    if mode not in ["face", "corners"]:
        print("Invalid mode. Defaulting to face tracking.")
        mode = "face"

    tracker = FaceTracking(mode=mode)
    tracker.input_type = input_type
    tracker.start()
