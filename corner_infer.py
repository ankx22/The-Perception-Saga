import cv2
import numpy as np


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# Load the image
filename = 'test_blob.png'
image_path = r"D:\WPI\Sem 3\Aerial vehicles\HW3\Corner detection\masks\washburn pics\masks\\" + filename
image = cv2.imread(image_path, 0)

  
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((3, 3), np.uint8) 
  
# The first parameter is the original image, 
# kernel is the matrix with which image is convolved and third parameter is the number 
# iterations will determine how much you want to erode/dilate a given image. 
img_erosion = cv2.erode(image, kernel, iterations=15) 
img_dilation = cv2.dilate(img_erosion, kernel, iterations=15) 


resized_image = resize_with_aspect_ratio(image, width=900) # You can adjust the width as needed
resized_image_erosion = resize_with_aspect_ratio(img_erosion, width=900) # You can adjust the width as needed
resized_image_dilation = resize_with_aspect_ratio(img_dilation, width=900) # You can adjust the width as needed

contours, _ = cv2.findContours(resized_image_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

resized_image_dilation_color = cv2.cvtColor(resized_image_dilation, cv2.COLOR_GRAY2BGR)

# print(contours)

for contour in contours:
    # Approximate polygon and ensure it has 4 corners
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # Draw circles on the corner points
        for point in approx:
            x,y = point[0]
            cv2.circle(resized_image_dilation_color, (int(x), int(y)), 7, (0, 0, 255), -1)
  
# cv2.imshow('Input', resized_image) 
# cv2.imshow('Erosion', resized_image_erosion) 
cv2.imshow('Dilation', resized_image_dilation_color) 
# cv2.imshow('Detected contours', img) 

# Save the image with corners marked
output_filename = 'marked_' + filename
output_path = r"D:\WPI\Sem 3\Aerial vehicles\HW3\Corner detection\masks\washburn pics\masks\\" + output_filename
print(output_path)
cv2.imwrite(output_path, resized_image_dilation_color)
  
cv2.waitKey(0) 
