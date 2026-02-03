import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import shutil
import argparse
import numpy as np

def compute_severity_score(row):
    """
    Composite severity score: higher = more severe.
    Adjust weights as needed.
    """
    return (
        row['lesion_percentage'] * 0.5 +
        row['lesion_contrast'] * 0.2 +
        row['lesion_hue_std_dev'] * 0.2 +
        (100 - row['mean_lesion_hue']) * 0.1
    )

def copy_images(filenames, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for filename in filenames:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1
    return count

def sort_by_severity_kmeans(features_file, image_source_dir, output_dir, n_clusters=3):
    print(f"\n--- Running Severity Sorting with K-Means ({n_clusters} clusters) ---")

    # Load features CSV
    try:
        df = pd.read_csv(features_file)
    except FileNotFoundError:
        print(f"Error: Features file '{features_file}' not found.")
        return

    # Prepare output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Copy healthy images as-is
    df_healthy = df[df['class'] == 'Pepper__bell___healthy'].copy()
    healthy_src = os.path.join(image_source_dir, 'Pepper__bell___healthy')
    healthy_dst = os.path.join(output_dir, '0_Healthy')
    healthy_count = copy_images(df_healthy['filename'], healthy_src, healthy_dst)
    print(f"✅ Copied {healthy_count} healthy images to '{healthy_dst}'")

    # Process diseased images
    df_disease = df[df['class'] != 'Pepper__bell___healthy'].copy()
    if df_disease.empty:
        print("No diseased images found to sort.")
        return

    # Compute severity scores
    df_disease['severity_score'] = df_disease.apply(compute_severity_score, axis=1)

    # Select features and scale
    feature_cols = ['lesion_percentage', 'mean_lesion_hue', 'lesion_contrast', 'lesion_hue_std_dev']
    X = df_disease[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_disease['cluster'] = kmeans.fit_predict(X_scaled)

    # Get cluster means by severity score and sort
    cluster_means = df_disease.groupby('cluster')['severity_score'].mean().sort_values()
    severity_labels = ['1_Early_Stage', '2_Mid_Stage', '3_Advanced_Stage'][:n_clusters]
    cluster_to_label = {cluster_id: severity_labels[i] for i, cluster_id in enumerate(cluster_means.index)}
    df_disease['severity_label'] = df_disease['cluster'].map(cluster_to_label)

    # Print cluster info
    print("\n📊 Cluster to Severity Mapping (based on mean severity score):")
    for cluster_id in cluster_means.index:
        count = (df_disease['cluster'] == cluster_id).sum()
        mean_score = cluster_means[cluster_id]
        label = cluster_to_label[cluster_id]
        print(f"  Cluster {cluster_id} → {label} | Images: {count} | Mean Score: {mean_score:.2f}")

    # Copy diseased images by severity label
    disease_src = os.path.join(image_source_dir, 'Pepper__bell___Bacterial_spot')
    for label in severity_labels:
        filenames = df_disease[df_disease['severity_label'] == label]['filename']
        dest_dir = os.path.join(output_dir, label)
        copied = copy_images(filenames, disease_src, dest_dir)
        print(f"📁 Copied {copied} diseased images to '{label}'")

    # Save CSV with assigned severity
    df_disease.to_csv(os.path.join(output_dir, 'diseased_severity_labeled.csv'), index=False)

    print(f"\n✅ Sorting complete. Images organized in '{output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort images by disease severity using K-Means clustering.")
    parser.add_argument('--features_file', type=str, default='disease_features.csv', help='Path to features CSV')
    parser.add_argument('--image_source_dir', type=str, default='dataset_segmented', help='Source image directory')
    parser.add_argument('--output_dir', type=str, default='Training_dataset_severity_sorted', help='Output folder')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters/severity levels')
    args = parser.parse_args()

    sort_by_severity_kmeans(args.features_file, args.image_source_dir, args.output_dir, args.clusters)
