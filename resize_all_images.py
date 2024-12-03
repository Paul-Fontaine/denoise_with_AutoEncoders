import os
import cv2

# resize all images in the datset to 256x256 in place

data_root = './SIDD_Small_sRGB_Only/Data'
resize_to = (256, 256)

i=0
for dir in os.listdir(data_root):
    for file in os.listdir(os.path.join(data_root, dir)):
        i += 1
        file_path = os.path.join(data_root, dir, file)
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, resize_to)
        cv2.imwrite(file_path, resized_image)
        print(f"Resized {file_path} to {resize_to}")

print("All images resized successfully")