from PIL import Image
import numpy as np
import os
import random

def load_images(dataset_path):
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    images = [Image.open(os.path.join(dataset_path, f)) for f in image_files]
    return images

def blend_images(image1, image2, alpha):
    blended_image = Image.blend(image1, image2, alpha)
    return blended_image

def save_augmented_image(image, output_path, index):
    image.save(os.path.join(output_path, f"augmented_{index}.jpg"))

def data_augmentation_mix(dataset_path, output_path, alpha_range=(0.3, 0.7)):
    images = load_images(dataset_path)
    num_images = len(images)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_images):
        # Randomly select another image from the dataset
        index = random.randint(0, num_images - 1)
        image1 = images[i]
        image2 = images[index]

        # Randomly choose an alpha value for blending
        alpha = np.random.uniform(*alpha_range)

        # Perform data augmentation by blending images
        augmented_image = blend_images(image1, image2, alpha)

        # Save the augmented image
        save_augmented_image(augmented_image, output_path, i)
    
def blend_roi_images(image1, image2, alpha, roi_box):
    # Create masks for blending
    mask1 = Image.new('L', image1.size, 0)
    mask2 = Image.new('L', image2.size, 0)
    mask1.paste(255, roi_box)
    mask2.paste(255, roi_box)

    # Blend only the specified ROI
    blended_image = Image.composite(image1, image2, mask1)

    # Blend the rest of the images
    blended_image.paste(Image.composite(image1, image2, mask2), (0, 0), mask2)

    return blended_image

def data_augmentation_blend_roi(dataset_path, output_path, alpha_range=(0.3, 0.7), roi_size=(600, 600)):
    images = load_images(dataset_path)
    num_images = len(images)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_images):
        # Randomly select another image from the dataset
        index = random.randint(0, num_images - 1)
        image1 = images[i]
        image2 = images[index]

        # Randomly choose an alpha value for blending
        alpha = np.random.uniform(*alpha_range)

        # Randomly choose a region of interest (ROI)
        width, height = image1.size
        x1 = random.randint(0, width - roi_size[0])
        y1 = random.randint(0, height - roi_size[1])
        roi_box = (x1, y1, x1 + roi_size[0], y1 + roi_size[1])

        # Perform data augmentation by blending only the specified ROI
        augmented_image = blend_roi_images(image1, image2, alpha, roi_box)

        # Save the augmented image
        save_augmented_image(augmented_image, output_path, i)


def main():
    dataset_path = 'data_sketches/train_images/000_airplane'
    output_path = 'data_sketches/blend/airplane'

    data_augmentation_blend_roi(dataset_path, output_path)

if __name__ == "__main__":
    main()
