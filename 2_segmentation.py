import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
from skimage.segmentation import chan_vese

VALID_EXTS = ['.jpg', '.jpeg', '.png']

def segment_leaf_final(args_tuple):
    """
    Implements the final, robust 4-stage segmentation logic to preserve the
    full leaf shape, including spots, while cleanly removing the background.
    """
    image_path, output_class_dir = args_tuple
    try:
        image = cv2.imread(image_path)
        if image is None: return

        # Prepare color spaces needed for the different stages
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segmentation_channel = lab_image[:, :, 1] # 'a*' channel for main algorithm

        # --- Stage 1: Combined Initial Mask (Green + Brown/Yellow) ---
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        lower_disease = np.array([10, 50, 20])
        upper_disease = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv_image, lower_disease, upper_disease)

        combined_mask = cv2.bitwise_or(green_mask, disease_mask)

        # --- Stage 2: Morphological Closing ---
        # Fills gaps to create a solid but tight initial shape
        kernel = np.ones((7,7), np.uint8)
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        initial_mask = np.zeros(segmentation_channel.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(initial_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # --- Stage 3: L*a*b* Chan-Vese ACS ---
        # Runs the precision algorithm on the high-contrast 'a*' channel
        cv_mask_bool = chan_vese(
            segmentation_channel, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
            max_num_iter=200, dt=0.5, init_level_set=initial_mask
        )

        # --- Stage 4: Final Contour Cleaning ---
        # Removes any leftover noise or halos using the "keep the biggest" rule
        intermediate_mask = cv_mask_bool.astype(np.uint8) * 255
        final_contours, _ = cv2.findContours(intermediate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros(segmentation_channel.shape, dtype=np.uint8)

        if final_contours:
            largest_final_contour = max(final_contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_final_contour], -1, 255, thickness=cv2.FILLED)

        # Apply the final, clean mask to the original image
        segmented_output = cv2.bitwise_and(image, image, mask=final_mask)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_class_dir, filename)
        cv2.imwrite(output_path, segmented_output)
    except Exception as e:
        print(f"[Error] Failed to process {image_path}: {str(e)}")

def run_segmentation(args):
    """Main function to find and process all images."""
    print("\n--- Running Module 2 : Leaf Segmentation ---")
    tasks = []
    for class_name in os.listdir(args.input_dir):
        input_class_dir = os.path.join(args.input_dir, class_name)
        if not os.path.isdir(input_class_dir): continue
        output_class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        image_paths = [p for p in glob.glob(os.path.join(input_class_dir, '*')) if os.path.splitext(p)[1].lower() in VALID_EXTS]
        for path in image_paths:
            tasks.append((path, output_class_dir))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(segment_leaf_final, tasks), total=len(tasks), desc="Segmenting Leaves"))
    print(f"\n--- Segmentation complete. Results saved to '{args.output_dir}' ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robustly segment leaves while preserving disease spots.")
    parser.add_argument('--input_dir', type=str, default='dataset_preprocessed', help='Path to the preprocessed dataset.')
    parser.add_argument('--output_dir', type=str, default='dataset_segmented', help='Path to save the final segmented images.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel processes.')
    args = parser.parse_args()
    run_segmentation(args)