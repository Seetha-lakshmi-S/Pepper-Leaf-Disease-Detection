import os
import shutil
from tqdm import tqdm

def prepare_data(source_dir, binary_dir, severity_dir):
    """
    Creates two new datasets:
    1. A binary (Healthy/Diseased) dataset for the 'GP' model.
    2. A 3-class (Early/Mid/Advanced) severity dataset for the 'Specialist' model.
    """
    print("\n--- Preparing Data for Two-Stage Models ---")

    # --- Create Binary Dataset --- 
    print(f"Creating binary dataset in '{binary_dir}'...")
    if os.path.exists(binary_dir): shutil.rmtree(binary_dir)
    
    # Copy the Healthy class
    healthy_src = os.path.join(source_dir, '0_Healthy')
    healthy_dst = os.path.join(binary_dir, 'Healthy')
    if os.path.exists(healthy_src):
        shutil.copytree(healthy_src, healthy_dst)

    # Combine all three disease stages into a single 'Diseased' class
    diseased_dst = os.path.join(binary_dir, 'Diseased')
    os.makedirs(diseased_dst)
    
    severity_classes = ['1_Early_Stage', '2_Mid_Stage', '3_Advanced_Stage']
    for stage in severity_classes:
        stage_src_path = os.path.join(source_dir, stage)
        if os.path.exists(stage_src_path):
            for filename in tqdm(os.listdir(stage_src_path), desc=f"Copying {stage} to Diseased"):
                shutil.copy(os.path.join(stage_src_path, filename), diseased_dst)

    # --- Create Severity Dataset (Diseased leaves only) ---
    print(f"\nCreating severity dataset in '{severity_dir}'...")
    if os.path.exists(severity_dir): shutil.rmtree(severity_dir)
    os.makedirs(severity_dir)

    for stage in tqdm(severity_classes, desc="Copying severity stages"):
        stage_src_path = os.path.join(source_dir, stage)
        stage_dst_path = os.path.join(severity_dir, stage)
        if os.path.exists(stage_src_path):
            shutil.copytree(stage_src_path, stage_dst_path)

    print("\n--- Data preparation complete. Two new datasets are ready. ---")

if __name__ == '__main__':
    SOURCE = 'Training_dataset_severity_sorted'
    BINARY_OUTPUT = 'binary_classifier_dataset'
    SEVERITY_OUTPUT = 'severity_classifier_dataset'
    prepare_data(SOURCE, BINARY_OUTPUT, SEVERITY_OUTPUT)