import cv2
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
from skimage.feature import graycomatrix, graycoprops

VALID_EXTS = ['.jpg', '.jpeg', '.png']

def extract_features_final(args_tuple):
    """
    Analyzes a segmented leaf to extract TCIFS features using the robust
    hybrid (HSV + Otsu) spot detection method.
    """
    image_path, class_name = args_tuple
    try:
        image = cv2.imread(image_path)
        if image is None: return None

        # --- Handle Healthy Leaves ---
        if 'healthy' in class_name.lower():
            return {
                'filename': os.path.basename(image_path),
                'class': class_name,
                'lesion_percentage': 0.0,
                'mean_lesion_hue': 0.0,
                'lesion_contrast': 0.0,
                'lesion_hue_std_dev': 0.0
            }

        # --- Process Diseased Leaves ---
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        total_leaf_pixels = np.count_nonzero(gray_image)
        if total_leaf_pixels == 0: return None

        # --- Hybrid Spot Detection Logic ---
        # 1. Fixed HSV Range
        lower_hsv = np.array([10, 50, 20])
        upper_hsv = np.array([30, 255, 200])
        mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # 2. Automated Otsu Thresholding
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        b_channel = lab_image[:, :, 2]
        _, mask_otsu = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_otsu = cv2.bitwise_and(mask_otsu, cv2.bitwise_not(green_mask))
        
        # 3. Combine Masks
        lesion_mask = cv2.bitwise_or(mask_hsv, mask_otsu)
        # --- End of Hybrid Logic ---
        
        lesion_pixels = np.count_nonzero(lesion_mask)

        # Initialize features
        lesion_percentage = (lesion_pixels / total_leaf_pixels) * 100 if total_leaf_pixels > 0 else 0
        mean_lesion_hue = 0.0
        lesion_hue_std = 0.0
        lesion_contrast = 0.0

        if lesion_pixels > 1:
            # Color Features
            lesion_hues = hsv_image[:, :, 0][lesion_mask > 0]
            mean_lesion_hue = np.mean(lesion_hues)
            lesion_hue_std = np.std(lesion_hues)

            # Texture Features
            lesion_texture_roi = np.zeros_like(gray_image)
            lesion_texture_roi[lesion_mask > 0] = gray_image[lesion_mask > 0]
            glcm = graycomatrix(lesion_texture_roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            lesion_contrast = graycoprops(glcm, 'contrast')[0, 0]

        return {
            'filename': os.path.basename(image_path),
            'class': class_name,
            'lesion_percentage': lesion_percentage,
            'mean_lesion_hue': mean_lesion_hue,
            'lesion_contrast': lesion_contrast,
            'lesion_hue_std_dev': lesion_hue_std
        }
    except Exception as e:
        print(f"[Error] Failed to process {image_path}: {str(e)}")
        return None

def run_feature_extraction(args):
    """Main function to find and process all images."""
    print("\n--- Running Module 3: Feature Extraction ---")
    tasks = []
    for class_name in os.listdir(args.input_dir):
        class_dir = os.path.join(args.input_dir, class_name)
        if not os.path.isdir(class_dir): continue
        image_paths = [p for p in glob(os.path.join(class_dir, '*')) if os.path.splitext(p)[1].lower() in VALID_EXTS]
        for path in image_paths:
            tasks.append((path, class_name))

    all_features = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(extract_features_final, tasks), total=len(tasks), desc="Extracting Features"))

    all_features = [res for res in results if res is not None]

    if not all_features:
        print("No features were extracted. Please check the input directory.")
        return

    df = pd.DataFrame(all_features)
    df.to_csv(args.output_csv, index=False)
    print(f"\n--- Feature extraction complete. Data saved to '{args.output_csv}' ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from segmented leaves using the hybrid spot detection method.")
    parser.add_argument('--input_dir', type=str, default='dataset_segmented', help='Path to the segmented dataset directory.')
    parser.add_argument('--output_csv', type=str, default='disease_features.csv', help='Path to save the final feature data.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel processes.')
    args = parser.parse_args()
    run_feature_extraction(args)