"""nao_visitor controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
# from controller import Robot

# create the Robot instance.

import time
import cv2
import numpy as np
from controller import Robot, Motion, Camera, GPS, Keyboard, Display
from transformers import BlipProcessor, TFBlipForConditionalGeneration
from PIL import Image

class Nao(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.timeStep = int(self.getBasicTimeStep())
        
        # Initialize devices
        self.findAndEnableDevices()
        self.loadMotionFiles()
        
        # Load BLIP model for comments
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.visitor_model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.setPath()

    def findAndEnableDevices(self):
        # Enable cameras
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraTop.enable(self.timeStep)
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraBottom.enable(self.timeStep)
        
        # Enable GPS
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timeStep)
        
        # Enable display
        self.display = self.getDevice("chest_display")
        
        # Enable keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timeStep)

    def loadMotionFiles(self):
        # Load the motion files
        self.forwards = Motion(r"D:\SHU\Robotics\assesment\motions\Forwards50.motion")
        self.backwards = Motion(r"D:\SHU\Robotics\assesment\motions\Backwards.motion")
        self.sideStepLeft = Motion(r"D:\SHU\Robotics\assesment\motions\SideStepLeft.motion")
        self.sideStepRight = Motion(r"D:\SHU\Robotics\assesment\motions\SideStepRight.motion")
        self.standInit = Motion(r"D:\SHU\Robotics\assesment\motions\StandInit.motion")

    def setPath(self):
        # Set predefined path locations (these should match your floor markings)
        self.path = [(0.5, 0.5), (2.0, 0.5), (3.0, 2.0)]  # Coordinates of waypoints in meters
        self.currentWaypoint = 0

    def moveToWaypoint(self, waypoint):
        x, y = waypoint
        current_position = self.gps.getValues()
        
        # Move robot towards the target (simplified path following)
        if x > current_position[0]:
            self.forwards.play()  # Play the motion to move forwards
        elif x < current_position[0]:
            self.backwards.play()  # Play the motion to move backwards
        
        if y > current_position[1]:
            self.sideStepLeft.play()  # Play the motion to move left
        elif y < current_position[1]:
            self.sideStepRight.play()  # Play the motion to move right

    def capture_photo(self):
        # Capture and save an image at the current location
        raw_img = self.cameraTop.getImageArray()
        img = np.array(raw_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img_path = f"photos/art_image.png"
        cv2.imwrite(img_path, img)
        return img_path

    def detect_red_arrow(self):
        # Load the red arrow template image
        red_arrow_template = cv2.imread("red_arrow.png", 0)  # Load as grayscale
        
        # Resize the template to ensure it's smaller than the camera image
        red_arrow_template = cv2.resize(red_arrow_template, (50, 50))  # Resize to 50x50 pixels (adjust as needed)
    
        # Capture image from the bottom camera
        raw_img = self.cameraBottom.getImageArray()
        if raw_img is None:
            return False  # No image captured
    
        img = np.array(raw_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
        # Convert to grayscale for template matching
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # Check dimensions before matching
        if red_arrow_template.shape[0] > gray_img.shape[0] or red_arrow_template.shape[1] > gray_img.shape[1]:
            print("Template is still larger than the camera image. Skipping detection.")
            return False
    
        # Perform template matching
        res = cv2.matchTemplate(gray_img, red_arrow_template, cv2.TM_CCOEFF_NORMED)
        
        threshold = 0.8
        loc = np.where(res >= threshold)
    
        return len(loc[0]) > 0  # True if match found # No match found

    def generate_comment(self, img_path):
        # Generate comment based on the artwork image using BLIP model
        image = Image.open(img_path)
        prompt = "Describe the painting: "
        inputs = self.processor(image, text=prompt, return_tensors="pt")
        
        out = self.visitor_model.generate(**inputs)
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
                img_path = self.capture_photo()
                comment = self.generate_comment(img_path)
                self.display_comment(comment)

                # Move to the next waypoint
                self.currentWaypoint += 1
                if self.currentWaypoint >= len(self.path):
                    break
            else:
                # Move robot towards the waypoint
                self.moveToWaypoint((x, y))
            
            # Check if the robot detects the red arrow mark on the floor
            if self.detect_red_arrow():
                # Stop robot and process the red arrow detection
                self.forwards.stop()  # Stop the current motion
                print("Red arrow detected. Stopping robot.")
                
                # Optionally, add time delay here while the robot processes the feedback
                time.sleep(2)  # Delay while processing feedback
                
                # After processing, continue moving to the next waypoint
                continue

            # Listen for keyboard input (optional control)
            key = self.keyboard.getKey()
            if key == ord('Q'):  # Quit if 'Q' is pressed
                break

    def run(self):
        # Start with initial standing motion
        self.standInit.play()  # Play the standing motion
        self.follow_path()

if __name__ == "__main__":
    robot = Nao()
    robot.run()