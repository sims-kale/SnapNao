from controller import Robot, Camera, Display, Motor, DistanceSensor
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from transformers import BlipProcessor, TFBlipForConditionalGeneration
import os


robot = Robot()
time_step = 64

# Initialize proximity sensors
ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(time_step)

# Devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")
camera.enable(time_step)
display = robot.getDevice("display")

# Set initial velocities
max_speed = 6.28

# Distance threshold for stopping (in meters)
stop_distance = 0.075  # 3 inches = 0.075 meters
distance_travelled = 0  # Track the distance travelled

# Stop duration for image capture
stop_duration = 20  # Stop for 20 seconds (time steps)

# Initialize flags
stopped = False
stop_counter = 0
comment = ""

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
visitor_model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def follow_wall():
    # Control robot to maintain distance from wall using distance sensors
    left_sensor_value = ps[0].getValue()  # Left sensor (ps0)
    right_sensor_value = ps[1].getValue()  # Right sensor (ps1)
    
    # Basic logic to make the robot walk straight
    threshold = 0.1  # For simplicity, we can define a threshold for wall distance
    if left_sensor_value < threshold:
        # Move slightly to the right to avoid getting too close to the wall
        left_motor.setVelocity(max_speed * 0.7)
        right_motor.setVelocity(max_speed)
    elif right_sensor_value < threshold:
        # Move slightly to the left to avoid getting too close to the wall
        left_motor.setVelocity(max_speed)
        right_motor.setVelocity(max_speed * 0.7)
    else:
        # Move straight
        left_motor.setVelocity(max_speed)
        right_motor.setVelocity(max_speed)

def rotate_left_90():
    # Rotate in place to the left for 90 degrees (counter-clockwise)
    for _ in range(15):  # ~90-degree turn
        left_motor.setVelocity(-0.5 * max_speed)
        right_motor.setVelocity(0.5 * max_speed)
        robot.step(time_step)
        
def rotate_right_90():
    # Rotate in place to the left for 90 degrees (counter-clockwise)
    for _ in range(15):  # ~90-degree turn
        left_motor.setVelocity(0.5 * max_speed)
        right_motor.setVelocity(-0.5 * max_speed)
        robot.step(time_step)

def capture_image():
    # Capture image from the camera
    image = camera.getImageArray()
    img_np = np.array(image, dtype=np.uint8)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("captured_image.jpg", frame)  # Save image as captured_image.jpg
    return frame

def generate_comment(image_path="captured_image.jpg"):
    # Generate a comment using the BLIP model
    image = Image.open(image_path).convert("RGB")
    prompt = "Wow, this is a beautiful painting because the painter's name is"
    
    # Convert image to numpy array (required by TensorFlow model)
    image_np = np.array(image)
    
    # Convert numpy array to TensorFlow tensor
    image_tf = tf.convert_to_tensor(image_np)
    
    # Prepare the model input as TensorFlow tensor
    inputs = procesor(images=image_tf, text=prompt, return_tensors="tf")
    
    # Generate output using the TensorFlow-based BLIP model
    out = visitor_model.generate(**inputs)
    
    # Decode the output and get the generated comment
    comment = processor.decode(out[0], skip_special_tokens=True)
    return comment

def display_comment(comment):
    # Create a black background for the display (RGBA format)
    img = np.zeros((display.getHeight(), display.getWidth(), 4), dtype=np.uint8)
    img_pil = Image.fromarray(img, mode="RGBA")
    cv2_img = np.array(img_pil)

    # Display the comment (truncate to 40 characters for clarity)
    cv2.putText(cv2_img, comment[:40], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1, cv2.LINE_AA)

    # Save the generated image to a temporary file
    temp_image_path = "temp_comment_image.png"
    cv2.imwrite(temp_image_path, cv2_img)

    # Load and display the image on the display
    if os.path.exists(temp_image_path):
        image_handle = display.imageLoad(temp_image_path)
        display.imagePaste(image_handle, 0, 0, False)
    else:
        display.setColor(0xFF0000)  # Set color to red
        display.drawText("Failed to load styled image", 50, 150)

    # Optionally, clean up by deleting the temporary image file
    os.remove(temp_image_path)

def run():
    global stopped, stop_counter, distance_travelled

    # Initialize the start time for stopping (for 2 seconds)
    stop_time = 0

    while robot.step(time_step) != -1:
        # If robot is stopped, handle stop duration
        if stopped:
            stop_counter += 1
            if stop_counter >= stop_duration:
                stopped = False
                stop_counter = 0
                distance_travelled = 0  # Reset distance after stopping
            else:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                continue

        # Stop after traveling for 2 seconds (assuming stop_time is the start of movement)
        if stop_time == 0:
            stop_time = robot.getTime()  # Get the current time (start of movement)

        # After 2 seconds, stop, turn left, capture the image, and generate the comment
        if robot.getTime() - stop_time >= 2:
            print("Stopping for image capture.")
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            rotate_left_90()  # Turn left to face the artwork
            frame = capture_image()  # Capture image
            comment = generate_comment()  # Generate comment from BLIP model
            print("Comment:", comment)  # Print the comment in the console
            display_comment(comment)  # Show the comment on the display
            rotate_right_90()  # Turn left again to continue straight
            stopped = True
            stop_time = 0  # Reset stop time
            continue

        # Continue moving straight (if not stopped)
        follow_wall()

        # Update distance traveled
        distance_travelled += max_speed * time_step / 1000.0  # Convert from mm to meters
run()
