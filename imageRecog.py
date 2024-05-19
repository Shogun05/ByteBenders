import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import os

class FaceRecognitionSystem:
    def __init__(self, encoded_images_dir, input_image_path, cropped_image_path, padding=20, desired_width=150, desired_height=150):
        self.encoded_images_dir = encoded_images_dir
        self.input_image_path = input_image_path
        self.cropped_image_path = cropped_image_path
        self.padding = padding
        self.desired_width = desired_width
        self.desired_height = desired_height

    def getImages(self):
        encoded_images_dir = self.encoded_images_dir
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images(self.encoded_images_dir)
        self.input_face_encoding, self.cropped_face_image = self.get_face_encoding_and_crop_image(self.input_image_path, self.cropped_image_path)
        self.run()
        
    def get_face_encoding_and_crop_image(self, image_path, save_path):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) > 0:
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            # Get the face location
            top, right, bottom, left = face_locations[0]
            # Apply padding
            top = max(0, top - self.padding)
            right = min(image.shape[1], right + self.padding)
            bottom = min(image.shape[0], bottom + self.padding)
            left = max(0, left - self.padding)
            # Crop the face from the image with padding
            face_image = image[top:bottom, left:right]
            
            # Ensure the image has proper dimensions
            face_image = cv2.resize(face_image, (self.desired_width, self.desired_height), interpolation=cv2.INTER_AREA)
            
            # Save the cropped face image
            cv2.imwrite(save_path, face_image)
            return face_encoding, face_image
        else:
            print("No face detected in the input image.")
            return None, None

    def run(self):
        if self.input_face_encoding is None:
            return
        
        # Display the cropped face image
        cv2.imshow("Cropped Face", self.cropped_face_image)

        # Load Camera
        cap = cv2.VideoCapture(0)

        flag = True
        while flag:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Detect Faces in the webcam feed
            face_locations, face_names = self.sfr.detect_known_faces(frame)

            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # Compare the detected face with the input face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_face_encoding = face_recognition.face_encodings(rgb_frame, [face_loc])[0]
                face_distances = face_recognition.face_distance([self.input_face_encoding], frame_face_encoding)
                confidence = 1 - face_distances[0]

                if confidence >= 0.5:
                    flag = False
                    color = (0, 255, 0)  # Green for verified
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                # Draw the name label
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                # Draw the "Verified" text below the bounding box if matched
                # if name == "Verified":
                #     cv2.putText(frame, "Verified", (x1, y2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

            # Display the frame with face annotations
            cv2.imshow("Frame", frame)

            # Break loop on 'ESC' key press
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Release the video capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()
