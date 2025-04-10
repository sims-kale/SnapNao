
import time
from controller import Robot, Keyboard, Motion, Display, Camera, GPS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
import os
import requests

class Nao(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.timeStep = int(self.getBasicTimeStep())
        
        # Initialize devices
        self.findAndEnableDevices()
        self.loadMotionFiles()
        
        # Load BLIP model for comments
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.setPath()

    def findAndEnableDevices(self):
        # Enable camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraTop.enable(self.timeStep)
        
        # Enable GPS
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timeStep)
        
        # Enable display
        self.display = self.getDevice("chest_display")
        
        # Enable keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timeStep)

    def loadMotionFiles(self):
        # Load Nao's movements for path navigation
        self.forwards = Motion("../../motions/Forwards50.motion")
        self.backwards = Motion("../../motions/Backwards.motion")
        self.sideStepLeft = Motion("../../motions/SideStepLeft.motion")
        self.sideStepRight = Motion("../../motions/SideStepRight.motion")
        self.standInit = Motion("../../motions/StandInit.motion")

    def setPath(self):
        # Set predefined path locations (these should match your floor markings)
        self.path = [(0.5, 0.5), (2.0, 0.5), (3.0, 2.0)]  # Coordinates of waypoints in meters
        self.currentWaypoint = 0

    def moveToWaypoint(self, waypoint):
        x, y = waypoint
        current_position = self.gps.getValues()
        
        # Move robot towards the target (simplified path following)
        if x > current_position[0]:
            self.startMotion(self.forwards)
        elif x < current_position[0]:
            self.startMotion(self.backwards)
        
        if y > current_position[1]:
            self.startMotion(self.sideStepLeft)
        elif y < current_position[1]:
            self.startMotion(self.sideStepRight)

    def capture_photo(self):
        # Capture and save an image at the current location
        raw_img = self.cameraTop.getImageArray()
        img = np.array(raw_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img_path = f"photos/art_image.png"
        cv2.imwrite(img_path, img)
        return img_path

    def generate_comment(self, img_path):
        # Generate comment based on the artwork image using BLIP model
        image = Image.open(img_path)
        prompt = "Describe the painting: "
        inputs = self.processor(image, text=prompt, return_tensors="pt")
        
        out = self.model.generate(**inputs)
        comment = self.processor.decode(out[0], skip_special_tokens=True)
        return comment

    def display_comment(self, comment):
        # Display the generated comment on Nao's chest display
        self.display.setColor(0x00FFFF)  # Cyan
        self.display.setFont("Arial", 14, True)
        self.display.drawText(comment, 10, 10)

    def follow_path(self):
        while self.step(self.timeStep) != -1:
            # Check if robot has reached the current waypoint
            current_position = self.gps.getValues()
            x, y = self.path[self.currentWaypoint]
            if abs(current_position[0] - x) < 0.1 and abs(current_position[1] - y) < 0.1:
                # Stop at the waypoint and capture image
                print(f"Reached waypoint {self.currentWaypoint + 1}")
                img_path = self.capture_photo()
                comment = self.generate_comment(img_path)
                self.display_comment(comment)

                # Move to the next waypoint
                self.currentWaypoint += 1
                if self.currentWaypoint >= len(self.path):
                    print("Reached all waypoints!")
                    break

            else:
                # Move robot towards the waypoint
                self.moveToWaypoint((x, y))

            # Listen for keyboard input (optional control)
            key = self.keyboard.getKey()
            if key == ord('Q'):  # Quit if 'Q' is pressed
                break

    def run(self):
        # Start with initial standing motion
        self.startMotion(self.standInit)
        self.follow_path()

if __name__ == "__main__":
    robot = Nao()
    robot.run()
