from controller import Robot, Keyboard, Motion, Display, Camera
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import sys
import random
# from snap_filters import apply_sunglasses_filter  #Deleted


class Nao(Robot):
    PHALANX_MAX = 8
    STYLES = {
        '1': 'vangogh.jpg',
        '2': 'picasso.jpg',
        '3': 'monet.jpg',
        '4': 'bauhaus.jpg',
        '5': 'abstract.jpg'
    }
    CURRENT_STYLE = 'vangogh.jpg'
    COMMENTS = [
        "A beautiful transformation into artistic expression, Thank you!",
        "This interpretation captures the essence of creativity, Thank you!",        
        "An elegant blend of technology and artistry. It looks Beautiful, Thank you!",        
        "The composition has wonderful balance and flow, Thank you!",        
        "You've inspired a truly remarkable piece, Thank you!",
        "Art meets AI. You're welcome, Thank you!",
        "Now this belongs in a gallery!, Thank you!",
        "What an interesting creative direction. Thank You!",
        "This composition draws the eye beautifully. Thank You",
        "A sophisticated interpretation of the original. Thank you",
        
        ]

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.timeStep = int(self.getBasicTimeStep())
        
        # Initialize devices
        self.findAndEnableDevices()
        # self.tts = self.getDevice("text_to_speech")
        self.speaker = self.getDevice("speaker")
        # self.speaker.speak("Testing sound.", 1.0)
        # self.speaker.playSound(self.speaker, self.speaker, "sounds.wav", 1.0, 1.0, 0.0, False)
        # sound_file = r"D:\SHU\Robotics\assesment\data\file_example_WAV_1MG.wav"
        # self.speaker.loadSound(sound_file)  # Preload
        # self.speaker.playSound(sound_file, 1.0, 1.0, 0.0, 0)
        self.loadMotionFiles()
        self.startMotion(self.standInit)
        
        # Load TF model
        print("Loading style transfer model...")
        self.style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        print("Model loaded successfully!")
        
        
        self.show_welcome_message()
        # self.print_style_menu()

    # robot sensors and actuators
    def findAndEnableDevices(self):
        
        # Camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraTop.enable(self.timeStep)
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraBottom.enable(self.timeStep)
        #display
        self.display = self.getDevice("chest_display")
        
        # Keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timeStep)
        
        # Motion sensors
        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(self.timeStep)

    def loadMotionFiles(self):
        self.standInit = Motion('../../motions/StandInit.motion')
        self.handWave = Motion('../../motions/HandWave.motion')
        self.forwards = Motion('../../motions/Forwards50.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')

    def show_welcome_message(self):
      
        self.startMotion(self.handWave)
        self.speaker.speak("Welcome to the SnapNao", 1.0)
        self.clear_display()
        self.display.setColor(0xFFFFFF)  # White text
        self.display.setFont("Arial", 14, True)
        self.display.drawText("Welcome to the Art Gallery!", 10, 0)
        self.display.drawText("Step into a masterpiece!\nPose, pick a style,\nand become part of art history!", 1, 30)
        self.display.setFont("Arial", 12, False)
        self.display.drawText("\n'C' - Capture photo", 1, 120)
        self.display.drawText("\n'H' - Home Screen", 1, 130)
        self.display.drawText("\n'Q' - Quite", 1, 150)

    def print_style_menu(self):
      
        style_text = f"Current Style: {self.CURRENT_STYLE.split('.')[0]}"
        self.display.setColor(0x00FF00)  # Green text
        self.display.setFont("Arial", 12, True)
        self.display.drawText(style_text, 1, 80)
        self.display.setFont("Arial", 12, False)
        self.display.drawText("1: Van Gogh\n2: Picasso\n3: Monet\n4: Bauhaus\n5: Abstract" , 50, 140)   #\nS: Sunglasses
        # self.display.drawText("2: Picasso", 50, 1)
        # self.display.drawText("3: Monet", 50, 1)

    def clear_display(self):
        self.display.setColor(0x000000)  # Black
        self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())

    # Capture and save a photo
    def capture_photo(self):
        
        self.speaker.speak("Pose and Smile please.", 1.0)
        raw_img = self.cameraTop.getImageArray()
        img = np.array(raw_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img_path = "photos/content_img.png"
        cv2.imwrite(img_path, img)
        # print(f"Captured image saved to {img_path}")
        self.display.setColor(0x00FFFF)
        self.display.setFont("Arial", 14, True)
        self.display.drawText("Image Captured!", 20, 20)
        self.speaker.speak("Image Captured!", 1.0)
        self.speaker.speak("Please Select the Style", 1.0)
        self.print_style_menu()
        

    # Apply selected style to saved image
    def apply_style_transfer(self, content_path, style_key):
    
        try:
            # Load content image
            content_img = cv2.imread(content_path)
            content_img = cv2.resize(content_img, (255, 255))
            content_tensor = tf.image.convert_image_dtype(content_img, tf.float32)[tf.newaxis, ...]
    
            # Load style image
            style_name = self.STYLES[style_key]
            style_path = os.path.join('styles', style_name)
            style_image = tf.image.decode_image(tf.io.read_file(style_path), channels=3)
            style_tensor = tf.image.convert_image_dtype(style_image, tf.float32)[tf.newaxis, ...]
    
            # Apply style
            stylized = self.style_model(content_tensor, style_tensor)[0]
            styled_img = np.array(stylized[0] * 255, dtype=np.uint8)
    
            # Save styled image
            styled_output_path = f"photos/styled_{style_name.split('.')[0]}.png"
            cv2.imwrite(styled_output_path, cv2.cvtColor(styled_img, cv2.COLOR_RGB2BGR))
            print(f"Styled image saved to {styled_output_path}")
            return styled_output_path

        except Exception as e:
            print(f"Error in style transfer: {e}")
            return None

    # def show_processing_message(self):
        # self.clear_display()
        # self.display.setColor(0x00FFFF)
        # self.display.setFont("Arial", 14, True)
        # self.display.drawText("Processing...", 80, 100)

    # Display the styled image from given path
    def display_styled_image(self, styled_path):
    
        if styled_path and os.path.exists(styled_path):
            self.clear_display()
            image_handle = self.display.imageLoad(styled_path)
            self.display.imagePaste(image_handle, 0, 0, False)
            style_comment = random.choice(self.COMMENTS)
            self.speaker.speak(style_comment, 1.0)
        else:
            self.display.setColor(0xFF0000)
            self.display.drawText("Failed to load styled image", 50, 150)
        
        
    #Change style and apply it to image

    def change_style(self, style_key):
        

        if style_key in self.STYLES:
            self.CURRENT_STYLE = self.STYLES[style_key]
            print(f"Changed style to: {self.CURRENT_STYLE}")
            content_path = "photos/content_img.png"
            if not os.path.exists(content_path):
                print("No captured image found. Press C to capture first.")
                return False
            styled_path = self.apply_style_transfer(content_path, style_key)
            self.display_styled_image(styled_path)
            return True
        return False
        
        
        
    # def apply_snap_filter(self, content_path, filter_type):
        # content_img = cv2.imread(content_path)
        # content_img = cv2.resize(content_img, (640,480))
        # cv2.imshow("content_img", content_img)
        # if filter_type == 'dog':
            # return apply_dog_filter(content_img)
        # if filter_type == 'sunglasses':
            # return apply_sunglasses_filter(content_img)
        # else:
            # return content_img





    def run(self):
    
        while self.step(self.timeStep) != -1:
            key = self.keyboard.getKey()
    
            if key == ord('C'):
                self.clear_display()
                self.capture_photo()
    
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                # self.speaker.speak("Your styled image is on the way.", 1.0)               
                self.change_style(chr(key))
    
            elif key == Keyboard.LEFT:
                self.startMotion(self.sideStepLeft)
            elif key == Keyboard.RIGHT:
                self.startMotion(self.sideStepRight)
            elif key == Keyboard.UP:
                self.startMotion(self.forwards)
            elif key == Keyboard.DOWN:
                self.startMotion(self.backwards)
                
            # elif key == ord('S'):
                # content_path = "photos/content_img.png"
                # styled_img = self.apply_snap_filter(content_path, filter_type='sunglasses')
                
                # if styled_img is not None:
                    # styled_img_rgb = cv2.cvtColor(styled_img, cv2.COLOR_BGR2RGB)
                    # self.display_styled_image(styled_img_rgb)
                    # cv2.imwrite("photos/styled_sunglasses.png", styled_img)
                # else:
                    # print("Failed to apply sunglasses filter.")
                
            elif key == ord('H'):
                self.show_welcome_message()
                # self.print_style_menu()
            elif key == ord('Q'):
                self.startMotion(self.handWave)
                break

    def startMotion(self, motion):
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()
        motion.play()
        self.currentlyPlaying = motion

if __name__ == "__main__":
   
    if not os.path.exists('styles'):
        os.makedirs('styles')
        print("Created 'styles' directory. Please add style images (e.g., vangogh.jpg)")

    robot = Nao()
    try:
        robot.run()
    except Exception as e:
        print(f"Error: {e}")
