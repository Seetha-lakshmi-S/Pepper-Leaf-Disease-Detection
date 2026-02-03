# Pepper Leaf Disease Detection using CNN-CapsNet 🌶️

### 📌 Project Overview

Agriculture faces significant threats from crop diseases, often resulting in massive yield losses. This project provides an automated, high-precision tool to identify **Pepper (Bell) Bacterial Spot** and **Healthy** leaves. By combining traditional Deep Learning with advanced Capsule Networks, this system offers more robust detection than standard models.

---

## 🚀 Key Highlights & Use Cases

* **Early Detection:** Helps farmers identify bacterial spots before they spread to the entire crop.
* **Reduced Pesticide Use:** By accurately identifying the specific disease, farmers can apply targeted treatments rather than broad-spectrum chemicals.
* **High Accuracy:** Leverages the **PlantVillage Dataset**, the gold standard in agricultural AI research.
* **User-Friendly Interface:** A Flask-based web dashboard allows anyone to upload a photo and get an instant diagnosis.

---

## 🧠 Technical Implementation: CNN + CapsNet

This project moves beyond standard Convolutional Neural Networks (CNNs) by integrating **Capsule Networks (CapsNet)**.

### 1. The CNN Layer (Feature Extraction)

The model uses initial Convolutional layers to act as local feature detectors. These layers identify basic patterns like the edges of a leaf and the specific yellow/brown tints of a bacterial lesion.

### 2. The CapsNet Advantage (Spatial Intelligence)

Standard CNNs use "Max Pooling," which can lose the exact location and orientation of features. We implemented **CapsNet** to solve this:

* **Equivariance:** Unlike CNNs which are "invariant" (they just see if a feature exists), CapsNet is "equivariant"—it understands the **position, size, and orientation** of the disease spots relative to the leaf.
* **Dynamic Routing:** The model uses an iterative routing-by-agreement mechanism, allowing capsules in one layer to "vote" for the correct prediction in the next layer, significantly reducing false positives.

---

## 🛠️ System Workflow

1. **Image Input:** User uploads a pepper leaf image via the Flask UI.
2. **Preprocessing & Segmentation:** The leaf is isolated from the background to remove noise.
3. **CNN-CapsNet Analysis:** The hybrid model processes the image to find spatial hierarchies of disease symptoms.
4. **Result Display:** The web app displays the classification (Healthy vs. Diseased) with a confidence score.

---

## 💻 How to Run Locally

1. **Clone the Repo:**
```bash
git clone https://github.com/Seetha-lakshmi-S/Pepper-Leaf-Disease-Detection.git

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Launch the App:**
```bash
python app.py

```


4. **Access:** Open `http://127.0.0.1:5000` in your browser.

---

