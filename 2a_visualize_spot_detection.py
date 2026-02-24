import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse

VALID_EXTS = ['.jpg', '.jpeg', '.png']

def visualize_spot_detection_hybrid(args_tuple):
    """
    Implements a hybrid spot detection method combining a fixed HSV range
    with automated Otsu thresholding for maximum accuracy and robustness.
    """
    image_path, class_name, out_dir_mask, out_dir_outline = args_tuple
    try:
        image = cv2.imread(image_path)
        if image is None or 'healthy' in class_name.lower():
            # Handle healthy leaves by creating blank outputs
            if image is not None:
                filename = os.path.basename(image_path)
                blank_mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.imwrite(os.path.join(out_dir_mask, filename), blank_mask)
                cv2.imwrite(os.path.join(out_dir_outline, filename), image)
            return

        # --- Method 1: Fixed HSV Range ---
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([10, 50, 20])
        upper_hsv = np.array([30, 255, 200])
        mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # --- Method 2: Automated Otsu Thresholding ---
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        b_channel = lab_image[:, :, 2]
        _, mask_otsu = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Refine Otsu mask by removing green areas
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_otsu = cv2.bitwise_and(mask_otsu, cv2.bitwise_not(green_mask))

        # --- Combine the results from both methods ---
        # A pixel is a spot if it's detected by HSV OR Otsu
        final_lesion_mask = cv2.bitwise_or(mask_hsv, mask_otsu)

        # Clean the final combined mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        filename = os.path.basename(image_path)
        
        # Save the visual outputs
        cv2.imwrite(os.path.join(out_dir_mask, filename), cleaned_mask)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlined_image = image.copy()
        cv2.drawContours(outlined_image, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir_outline, filename), outlined_image)

    except Exception as e:
        print(f"[Error] Failed to process {image_path}: {str(e)}")

def run_visualization(args):
    print("\n--- Running Hybrid Spot Detection Visualization ---")
    output_mask_dir = os.path.join(args.output_base_dir, 'spot_masks')
    output_outline_dir = os.path.join(args.output_base_dir, 'spots_outlined')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_outline_dir, exist_ok=True)

    tasks = []
    for class_name in os.listdir(args.input_dir):
        class_dir = os.path.join(args.input_dir, class_name)
        if not os.path.isdir(class_dir): continue
        os.makedirs(os.path.join(output_mask_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_outline_dir, class_name), exist_ok=True)
        
        image_paths = [p for p in glob.glob(os.path.join(class_dir, '*')) if os.path.splitext(p)[1].lower() in VALID_EXTS]
        for path in image_paths:
            tasks.append((path, class_name, os.path.join(output_mask_dir, class_name), os.path.join(output_outline_dir, class_name)))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(visualize_spot_detection_hybrid, tasks), total=len(tasks), desc="Visualizing Spots"))
    print(f"\n--- Visualization complete. Results saved to '{args.output_base_dir}' ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize spot detection using a hybrid HSV and Otsu method.")
    parser.add_argument('--input_dir', type=str, default='dataset_segmented', help='Path to the segmented dataset directory.')
    parser.add_argument('--output_base_dir', type=str, default='verification_outputs', help='Base path to save the visualization outputs.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel processes.')
    args = parser.parse_args()
    run_visualization(args)