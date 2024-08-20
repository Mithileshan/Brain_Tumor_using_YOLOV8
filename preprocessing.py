import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, rotation_angle=0, translation=(0, 0), grayscale=False, brightness=1.0, exposure=1.0):
    img = cv2.imread(image_path)

    if rotation_angle != 0:
        rows, cols, _ = img.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    if translation != (0, 0):
        translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        img = cv2.warpAffine(img, translation_matrix, (cols, rows))

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
    img = cv2.convertScaleAbs(img, alpha=exposure, beta=0)

    return img

def preprocess_images_in_folder(input_folder, output_folder, rotation_angle=0, translation=(0, 0), grayscale=False, brightness=1.0, exposure=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_images = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            processed_image = preprocess_image(input_path, rotation_angle, translation, grayscale, brightness, exposure)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

            processed_images.append(processed_image)

    return processed_images

def visualize_random_images(images, num_images=5):
    if not images:
        print("No processed images found.")
        return

    if len(images) < num_images:
        print(f"Warning: Number of processed images ({len(images)}) is less than the specified num_images ({num_images}).")

    flattened_images = [img.flatten() for img in images]
    selected_indices = np.random.choice(len(flattened_images), min(num_images, len(images)), replace=False)

    for index in selected_indices:
        img = flattened_images[index].reshape(images[0].shape)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else 'viridis')
        plt.title("Processed Image")
        plt.axis('off')
        plt.show()
# Example usage:
input_folder = 'train/images/'
output_folder = 'processed/'

processed_images = preprocess_images_in_folder(input_folder, output_folder, rotation_angle=45, translation=(20, 10), grayscale=True, brightness=1.2, exposure=1.5)

# Print the number of processed images


# Visualize random images
visualize_random_images(processed_images)