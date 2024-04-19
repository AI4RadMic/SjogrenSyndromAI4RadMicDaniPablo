

import cv2
import numpy as np
import os

def process_image(img_path):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    cropped_img = img[:350, 150:500]


    return cropped_img

def process_images_in_folder(input_folder, output_folder):
    """
    Crop and resize images from input_folder and save them to output_folder.
    
    Parameters:
    - input_folder: Path to the folder containing the original images.
    - output_folder: Path to the folder where the modified images will be saved.
    - crop_area: A tuple (left, top, right, bottom) for the cropping area.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            processed_img = process_image(img_path)
            
            # Save the image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)

            print(f"Processed and saved {filename}")
            
        

'''# Example usage
input_folder = 'data/images'
output_folder = 'data/processed_images'
process_images_in_folder(input_folder, output_folder)'''
