# Package-Detection-Model
A detection model meant to announce package deliveries by sending an email including the type of package and photo of the package.

![Example of image annotated by package detection model.][(https://drive.google.com/file/d/1ZIOUkvd5iO3DLFdVguLkTyVH_iWQ-W2I/view?usp=sharing)]

## My Process
To begin I compiled a large dataset (over 2000 images) by manually finding images of my three classes, Amazon, FedEx, and Other boxes, resizing them using Resize.py in Visiual Studio Code and annotating them individually using Roboflow. I then used yolo_model_train.py in VSCode to train the model using YOLOv8 (I only went to 50 epochs which took just under two hours). After training the model I created img_vid_live_test.py to test how well the model worked and once I was satisfied I modified img_vid_live_test.py to Email me when it detects a package. I have several changes I hope to add such as adding a package counter/logic system so I don't get an email every frame, make the program work for live video by using NoMachine to SSH into my Nvidia Jetson Orin Nano and tweaking img_vid_live_test.py to accomodate, and changing the email bot so it includes the class/type of package in the Email. I might also increase the size of the dataset to more thoroughly train the model. Once finished, I will be able to hook it up to my doorbell camera, and have it Email me whenever a **new** package arrives.

## Running this project
1. In VSCode, install Pytorch 2.1.0, Torchvision 0.16.0, Numpy 1.23.0, Ultralytics, and Roboflow
2. Download the best.pt model and img_vid_live_test.py into VSCode
3. Update img_vid_live_test.py with the best.pt model path
4. In order to test the email feature, you would need a gmail account with an app password, but once you have one, update the global variables at the top with your sender email, sender app password, and recipient email
5. Without the email feature img_vid_live_test.py still works to detect and annotate photos and videos, just run the code, select "1" for image mode and "2" for video mode, and paste in the image/video path
6. The annotated image/video should save in the same location as the original

[View a video demonstration here](https://drive.google.com/file/d/1_zWDO7wkwc1nvBqYkR9BtnZFy5XKsclp/view?usp=sharing)
Planning Document: https://docs.google.com/document/d/1X50kOJL4-UMfRQnMR9EwE3PvWpGpUIG0t8-N_gPw_HI/edit?usp=sharing 
