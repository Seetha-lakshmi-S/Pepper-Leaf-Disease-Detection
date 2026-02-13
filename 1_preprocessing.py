import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse

# Allowed image extensions
VALID_EXTS = ['.jpg', '.jpeg', '.png']

def process_image(args_tuple):
    """Worker function to process a single image."""
    image_path, output_class_dir, resize, target_size, ksize = args_tuple
    try:
        image = cv2.imread(image_path)

        if image is None:
            # Using a lock for thread-safe printing
            with threading.Lock():
                print(f"[Warning] Skipped unreadable image: {image_path}")
            return

        if resize:
            image = cv2.resize(image, target_size)

        filtered_image = cv2.medianBlur(image, ksize)

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_class_dir, filename)
        cv2.imwrite(output_path, filtered_image)

    except Exception as e:
        with threading.Lock():
            print(f"[Error] Failed to process {image_path}: {str(e)}")

def run_preprocessing(args):
    """Main function to find and process all images based on CLI arguments."""
    print("\n--- Running Module 1: Preprocessing ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all class subdirectories
    class_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    tasks = []
    for class_name in class_dirs:
        input_class_dir = os.path.join(args.input_dir, class_name)
        output_class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        image_paths = [p for p in glob.glob(os.path.join(input_class_dir, '*')) if os.path.splitext(p)[1].lower() in VALID_EXTS]
        
        for path in image_paths:
            tasks.append((path, output_class_dir, args.resize, (args.width, args.height), args.ksize))

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Filtering Images"))

    print(f"\n--- Preprocessing complete. Results saved to '{args.output_dir}' ---")

if __name__ == '__main__':
    # --- Command-Line Interface Setup ---
    parser = argparse.ArgumentParser(description="Preprocess image dataset for plant disease classification.")
    parser.add_argument('--input_dir', type=str, default='dataset', help='Path to the original dataset directory.')
    parser.add_argument('--output_dir', type=str, default='dataset_preprocessed', help='Path to save the preprocessed images.')
    parser.add_argument('--ksize', type=int, default=5, help='Kernel size for the Median Filter (must be odd).')
    parser.add_argument('--resize', action='store_true', help='Add this flag to resize images.')
    parser.add_argument('--width', type=int, default=256, help='Target width for resizing.')
    parser.add_argument('--height', type=int, default=256, help='Target height for resizing.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel threads to use.')
    
    args = parser.parse_args()

    # Ensure ksize is an odd number
    if args.ksize % 2 == 0:
        args.ksize += 1
        print(f"Warning: Kernel size must be odd. Adjusting to {args.ksize}")

    run_preprocessing(args)