import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Function to detect face shape
def detect_face_shape(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
        return

    face_landmarks = results.multi_face_landmarks[0].landmark

    # Key facial points
    forehead_top = face_landmarks[10]  
    chin_bottom = face_landmarks[152]  
    left_cheekbone = face_landmarks[234]  
    right_cheekbone = face_landmarks[454]  
    left_jaw = face_landmarks[172]  
    right_jaw = face_landmarks[397]  

    # Measurements
    face_width = calculate_distance(left_cheekbone, right_cheekbone)
    jaw_width = calculate_distance(left_jaw, right_jaw)
    face_height = calculate_distance(forehead_top, chin_bottom)

    # Determine face shape
    if face_height > face_width * 1.3:
        face_shape = "Oval"
    elif face_width >= face_height * 0.9:
        face_shape = "Round"
    elif jaw_width < face_width * 0.8:
        face_shape = "Heart"
    elif abs(face_width - jaw_width) < face_width * 0.1:
        face_shape = "Square"
    else:
        face_shape = "Diamond"

    print(f"Face Shape: {face_shape}")

    # Draw face landmarks
    annotated_image = image.copy()
    for landmark in [forehead_top, chin_bottom, left_cheekbone, right_cheekbone, left_jaw, right_jaw]:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow("Face Shape Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Shape Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face Shape Detection", annotated_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# Function to classify skin color
def classify_skin_color(image):
    hsvconvt = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 87])
    upper_skin = np.array([26, 160, 210])
    mask = cv2.inRange(hsvconvt, lower_skin, upper_skin)
    skinv = hsvconvt[mask > 0, 2]
    avg = np.mean(skinv)
    
    if 120 < avg < 135:
        skin_color = "Brown skin"
    elif 135 < avg < 140:
        skin_color = "Light brown skin"
    elif avg > 140:
        skin_color = "Light skin"
    elif avg < 120:
        skin_color = "Dark skin"
    
    print(f"Skin Color: {skin_color}")

    #cv2.namedWindow("Skin Mask", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("Skin Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("Skin Mask", mask)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()

# Function to measure body shape
def measure_body_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No pose detected in the image.")
        return

    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    hip_width = calculate_distance(left_hip, right_hip)
    shr = shoulder_width / hip_width

    print(f"Body Type: {'V-shaped' if shr >= 1.4 else 'Balanced' if shr >= 0.9 else 'Uneven'}")

    annotated_image = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.namedWindow("Body Shape Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Body Shape Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Body Shape Detection", annotated_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# Initialize camera
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
camera.release()

cv2.imwrite("screenshot.png", frame)
image = cv2.imread("screenshot.png")

# Display "Put on Your Model Face" message
face = np.ones((300, 600, 3), dtype=np.uint8)
cv2.putText(face, "Put on Your Model Face", (100, 150),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (255, 0, 255), 2, cv2.LINE_AA)

cv2.namedWindow("Put on Your Model Face", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Put on Your Model Face", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Put on Your Model Face", face)
cv2.waitKey(4000)
cv2.destroyAllWindows()

detect_face_shape(image)

# Compliments list
import random
import cv2
import numpy as np

# Compliments list with manually split lines
import random
import cv2
import numpy as np

# Compliments list with manually split lines
import random
import cv2
import numpy as np

# Compliments list with manual line breaks
compliments = [
    "Your face could launch\n a thousand smiles!",
    "If there were a beauty contest\n for awesome faces,\n you would win every time!",
    "Your smile must be a\n cheat code for happiness!",
    "Your face has 'main character\n energy' written all over it!",
    "I think your face just\n upgraded the entire room's aesthetic!"
]

# Pick a random compliment
h = random.choice(compliments)

# Get screen resolution (modify if needed)
screen_width = 1920
screen_height = 1080

# Create a blank white image
nice = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

# Split the text into multiple lines
lines = h.split("\n")

# Font settings
font_scale = 3
thickness = 5
line_spacing = 120  # Increased spacing between lines

# Calculate dynamic starting y-position to center the text properly
text_height = len(lines) * line_spacing  # Total height occupied by text
y_offset = (screen_height - text_height) // 3  # Start from a balanced position (1/3 from top)

# Draw compliment text
for line in lines:
    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, thickness)[0]
    x_pos = (screen_width - text_size[0]) // 2  # Center align horizontally
    cv2.putText(nice, line, (x_pos, y_offset),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    y_offset += line_spacing  # Move down for the next line

# Add a big smiley face at the bottom center
smiley_center = (screen_width // 2, screen_height - 250)  # Position at bottom center
smiley_radius = 100

# Draw smiley face
cv2.circle(nice, smiley_center, smiley_radius, (0, 255, 255), -1)  # Face (Yellow)
cv2.circle(nice, (smiley_center[0] - 30, smiley_center[1] - 30), 15, (0, 0, 0), -1)  # Left Eye (Black)
cv2.circle(nice, (smiley_center[0] + 30, smiley_center[1] - 30), 15, (0, 0, 0), -1)  # Right Eye (Black)
cv2.ellipse(nice, (smiley_center[0], smiley_center[1] + 20), (40, 20), 0, 0, 180, (0, 0, 0), 5)  # Smile

# Show the compliment window in full screen
cv2.namedWindow("Compliment", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Compliment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Compliment", nice)
cv2.waitKey(6000)
cv2.destroyWindow("Compliment")


classify_skin_color(image)

# Warning to move back
warning_img = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(warning_img, "Move 2 feet back and stand!", (100, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

cv2.namedWindow("Move Back Warning", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Move Back Warning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Move Back Warning", warning_img)
cv2.waitKey(6000)
cv2.destroyAllWindows()

measure_body_from_image(image)

dress_recommendations = {
    ("Fit (V-shaped body)", "Light skin"): ("Slim-fit blazer & trousers", r"C:\Users\harsh\recommendations\dresses\fit_light.jpg"),
    ("Fit (V-shaped body)", "Dark skin"): ("Dark-tone tailored suit", r"C:\Users\harsh\recommendations\dresses\fit_dark.jpg"),
    ("Balanced body", "Light skin"): ("A-line dresses & pastel tones", r"C:\Users\harsh\recommendations\dresses\balanced_light.jpg"),
    ("Balanced body", "Dark skin"): ("Earth-tone wrap dresses", r"C:\Users\harsh\recommendations\dresses\balanced_dark.jpg"),
    ("Thin or Fat (Uneven body proportions)", "Light skin"): ("Empire waist dresses", r"C:\Users\harsh\recommendations\dresses\thin_light.jpg"),
    ("Thin or Fat (Uneven body proportions)", "Dark skin"): ("Layered outfits with warm tones", r"C:\Users\harsh\recommendations\dresses\thin_dark.jpg")
}

haircut_recommendations = {
    "Oval": ("Medium-length layers", r"C:\Users\harsh\recommendations\haircuts\oval_cut.jpg"),
    "Round": ("Pompadour & volume on top", r"C:\Users\harsh\recommendations\haircuts\round_cut.jpg"),
    "Square": ("Soft waves & layered styles", r"C:\Users\harsh\recommendations\haircuts\square_cut.jpg"),
    "Heart": ("Side-swept bangs & chin-length cuts", r"C:\Users\harsh\recommendations\haircuts\heart_cut.jpg"),
    "Diamond": ("Short textured cuts or long layers", r"C:\Users\harsh\recommendations\haircuts\diamond_cut.jpg")
}

def display_recommendation(title, recommendation, image_path):
    """
    Displays the recommended dress or haircut in full screen.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Get screen resolution
    screen_width = 1920
    screen_height = 1080

    # Resize image to fit full screen
    image = cv2.resize(image, (screen_width, screen_height))

    # Overlay text on image
    cv2.putText(image, f"Recommended: {recommendation}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)

    # Display full-screen window
    cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(title, image)
    cv2.waitKey(8000)
    cv2.destroyWindow(title)

def recommend_outfit_and_haircut(body_type, skin_tone, face_shape):
    """
    Recommends an outfit based on body shape & skin tone, and a haircut based on face shape.
    """
    # Recommend dress
    dress_key = (body_type, skin_tone)
    if dress_key in dress_recommendations:
        dress_name, dress_image = dress_recommendations[dress_key]
        display_recommendation("Dress Recommendation", dress_name, dress_image)
    else:
        print("No dress recommendation available.")

    # Recommend haircut
    if face_shape in haircut_recommendations:
        haircut_name, haircut_image = haircut_recommendations[face_shape]
        display_recommendation("Haircut Recommendation", haircut_name, haircut_image)
    else:
        print("No haircut recommendation available.")

# Example usage (replace with actual detected values)
detected_body_type = "Balanced body"  # Example detected body type
detected_skin_tone = "Dark skin"  # Example detected skin tone
detected_face_shape = "Heart"  # Example detected face shape

recommend_outfit_and_haircut(detected_body_type, detected_skin_tone, detected_face_shape)

