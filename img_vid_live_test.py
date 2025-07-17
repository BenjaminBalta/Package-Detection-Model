#add package counter to restrict email sending
#make it work for live video
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import nms_patch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

SENDER_EMAIL = "NvidiaPackageBot@gmail.com"
SENDER_PASSWORD = "eyvo mufy vyvc ndrl"
RECIPIENT_EMAIL = "NvidiaPackageBot@gmail.com"
SUBJECT = "Package Detected At Front Door"

def detect_objects_image(model_path, image_path, conf_threshold=0.5, save_results=True):
    """
    Detect objects in an image using YOLOv8 model

    Args:
        model_path (str): Path to your yolov8.pt model file
        image_path (str): Path to the input image
        conf_threshold (float): Confidence threshold for detections
        save_results (bool): Whether to save annotated image

    Returns:
        results: YOLO detection results
    """

    # Load the model
    model = YOLO(model_path)

    # Run inference
    results = model(image_path, conf=conf_threshold)

    # Process results
    for r in results:
        # Get image dimensions
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get boxes, classes, and confidence scores
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        print(f"Found {len(boxes)} objects:")
        print("-" * 50)

        # Draw bounding boxes and labels
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            class_name = model.names[int(cls)]
            print(f"Object {i+1}: {class_name} (confidence: {conf:.2f})")
            print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")

            # Draw rectangle
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_rgb, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'YOLOv8 Object Detection - {len(boxes)} objects found')
    plt.show()

    email_body = f'{len(boxes)} package detected'

    # Save annotated image
    if save_results:
        output_path = image_path.replace('.', '_detected.')
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"\nAnnotated image saved as: {output_path}")
 
    send_photo(sender_email = SENDER_EMAIL, sender_password = SENDER_PASSWORD, recipient_email = RECIPIENT_EMAIL, subject = SUBJECT, body = email_body, photo_path = output_path)
    return results

def detect_objects_video(model_path, video_path, conf_threshold=0.5, save_video=True, display_video=True):
    """
    Detect objects in a video using YOLOv8 model
    
    Args:
        model_path (str): Path to your yolov8.pt model file
        video_path (str): Path to input video file
        conf_threshold (float): Confidence threshold for detections
        save_video (bool): Whether to save output video
        display_video (bool): Whether to display video in real-time

    Returns:
        None
    """
    # Load the model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")

    # Setup video writer if saving
    if save_video:
        output_path = video_path.replace('.', '_detected.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved as: {output_path}")

    # Process video frame by frame
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run YOLOv8 inference on frame
        results = model(frame, conf=conf_threshold)

        # Process results
        annotated_frame = frame.copy()

        for r in results:
            # Get detection data
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            # Draw bounding boxes and labels
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                class_name = model.names[int(cls)]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label with background
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1-10), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
        # Display frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(boxes)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save frame to output video
        if save_video:
            out.write(annotated_frame)

        # Display frame
        if display_video:
            cv2.imshow('YOLOv8 Video Detection', annotated_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection stopped by user")
                break
                
        # Show progress
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed
            print(f"Processing frame {frame_count}/{total_frames} | FPS: {fps_current:.2f}")

    # Cleanup
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"\nVideo processing completed!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}")

def detect_objects_webcam(model_path, conf_threshold=0.5, save_video=False):
    """
    Detect objects from webcam feed using YOLOv8 model
    
    Args:
        model_path (str): Path to your yolov8.pt model file
        conf_threshold (float): Confidence threshold for detections
        save_video (bool): Whether to save webcam output

    Returns:
        None
    """

    # Load the model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0) # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Webcam properties: {width}x{height} @ {fps} FPS")
    print("Press 'q' to quit, 's' to save current frame")

    # Setup video writer if saving
    if save_video:
        output_path = f"webcam_detection_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        print(f"Recording to: {output_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold)

        # Process results
        annotated_frame = frame.copy()

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            # Draw detections
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                class_name = model.names[int(cls)]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display info
        info_text = f"Objects: {len(boxes)} | Frame: {frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame if recording
        if save_video:
            out.write(annotated_frame)
            
        # Display frame
        cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            save_path = f"webcam_frame_{int(time.time())}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Frame saved as: {save_path}")

    # Cleanup
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()


def send_photo(sender_email,sender_password,recipient_email,subject,body,photo_path,smtp_server="smtp.gmail.com",smtp_port=587):

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body,'plain'))

    if not os.path.exists(photo_path):
        raise FileNotFoundError(f'Photo file not found: {photo_path}')

    with open(photo_path,'rb') as attachment:
        img = MIMEImage(attachment.read())
        filename = os.path.basename(photo_path)
        img.add_header('Content-Disposition', f'attachment; filename= {filename}')
        msg.attach(img)
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email,sender_password)
        text = msg.as_string()
        server.sendmail(sender_email,recipient_email,text)
        server.quit()

    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual paths
    MODEL_PATH = "/home/nvidia/Package-Detection-Model/best.pt" # Path to your model
    
    # Choose detection mode
    print("YOLOv8 Detection Options:")
    print("1. Image detection")
    print("2. Video file detection")
    print("3. Webcam detection")
    
    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        IMAGE_PATH = input("Enter image path: ") # Replace with your image path
        detect_objects_image(MODEL_PATH, IMAGE_PATH)
        
    elif choice == "2":
        VIDEO_PATH = input("Enter video path: ") # Replace with your video path
        detect_objects_video(MODEL_PATH, VIDEO_PATH, 
                             conf_threshold=0.5,
                             save_video=True,
                             display_video=True)
                             
    elif choice == "3":
        detect_objects_webcam(MODEL_PATH,
                              conf_threshold=0.5,
                              save_video=False)
                              
    else:
        print("Invalid choice!")