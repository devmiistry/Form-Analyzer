import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

# Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle calculation
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Correct video orientation (for MOV and other formats)
def correct_video_orientation(input_path, output_path):
    clip = VideoFileClip(input_path)
    # Forces moviepy to consider rotation metadata
    clip = clip.fx(vfx.rotate, 0)  
    clip.write_videofile(output_path, codec='libx264', audio=False)

# Input and Output paths
input_video_path = 'IMG_0742.MOV'  # Replace with your filename
corrected_video_path = 'corrected_video.mp4'  # Path for corrected video

# Step 1: Correct video orientation
correct_video_orientation(input_video_path, corrected_video_path)

# Step 2: Open the corrected video file
cap = cv2.VideoCapture(corrected_video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the original width and height
        height, width, _ = frame.shape

        # Resize the frame to maintain aspect ratio (this scales to 640px width, adjusting height accordingly)
        new_width = 640
        new_height = int(new_width * height / width)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose
        results = pose.process(image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Calculate hip angle
            hip_angle = calculate_angle(shoulder, hip, knee)

            # Display angle
            cv2.putText(image, str(int(hip_angle)),
                        tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback logic
            if hip_angle < 140:
                feedback = "Hinge deeper!"
            elif hip_angle > 170:
                feedback = "Stand straighter"
            else:
                feedback = "Good posture"

            # Show feedback
            cv2.putText(image, feedback, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        except:
            pass

        # Render pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

        # Show result
        cv2.imshow('Exercise Form Analysis', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
