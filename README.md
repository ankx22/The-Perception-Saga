# Mini Drone Race – The Perception Saga!



## Introduction
This project aims to develop a comprehensive autonomy stack that enables a quadrotor to navigate through multiple windows, drawing inspiration from the AlphaPilot competition. The project is divided into two main parts: the perception stack and the planning, control, and integration stack. The ultimate goal is to deploy these stacks on a real quadrotor, achieving robust autonomous navigation in complex environments.

(Check the full problem statements here [project3a](https://rbe549.github.io/rbe595/fall2023/proj/p3a/) and [project3b](https://rbe549.github.io/rbe595/fall2023/proj/p3b/))


## Steps to run the code
- Install Numpy, OpenCV, djitellopy, Ultralytics libraries before running the code.
- To train the YOLOv8 model run the `window_seg_yolov8.ipynb` file and add the location of your data accordingly.
- To do data augmentation on the images obtained of the windows using blender you can use the `image_warp.py` file and generate augmented images.
- You also need to perform camera calibration for your specific drone. This can be achieved using the `calibration.py` file and images can be taken on the DJI Tello drone using the `calibimage.py` file.
- There is also a `corner_infer.py` file to test your model to infer the corners of any image you want to test for.
- To run the main code run the `fly_tello.py` file after installing all dependancies and specifying the correct path to your trained model weights.
- In Code folder:
  ```bash
  python3 fly_tello.py
  ```


## Report
For detailed description of the math see the report [here](Report.pdf).## Collaborators
Ankit Talele - amtalele@wpi.edu

Chaitanya Sriram Gaddipati - cgaddipati@wpi.edu

Shiva Surya Lolla - slolla@wpi.edu


  
