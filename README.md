# 🧠 Brain Tumor Detection and Classification

## 📌 Project Overview

This project focuses on **detecting and classifying brain tumors from MRI images using deep learning models**. The system allows users to upload an MRI scan through a web interface, after which the trained model automatically analyzes the image and predicts the tumor type.

The model classifies MRI images into **four categories**:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

In addition to classification, the system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations of the model’s predictions. Grad-CAM highlights the important regions in the MRI image that influenced the model’s decision, improving **model interpretability and transparency**.

The application is built using **Flask** for the web interface and **TensorFlow / Keras** for implementing and running the deep learning models.

---

## 🧪 Models Experimented

During development, multiple deep learning architectures were tested:

1. **ResNet**
2. **EfficientNet**
3. **Hybrid Model (ResNet + EfficientNetB7)**

After experimentation, the **Hybrid ResNet + EfficientNetB7 model** provided better performance and was selected as the **final model**.

---

## ⚙️ How the System Works

1. User uploads an **MRI brain image** through the web interface.
2. The image is **preprocessed and resized**.
3. The trained **Hybrid CNN model** predicts the tumor category.
4. The system displays the predicted class:

   * Glioma
   * Meningioma
   * Pituitary
   * No Tumor
5. **Grad-CAM visualization** highlights the important region of the MRI image used by the model for prediction.

---

## 📓 Model Training Notebooks

The deep learning models used in this project were trained using **Google Colab notebooks**.

### 🔗 ResNet50 Model
Training notebook for the **ResNet50 architecture**:

[Open ResNet50 Training Notebook](https://colab.research.google.com/drive/1xg1bfN8_3QndLAjqOvtErIGUn5IgM4RY?usp=sharing)

---

### 🔗 EfficientNetB7 Model
Training notebook for the **EfficientNetB7 architecture**:

[Open EfficientNetB7 Training Notebook](https://colab.research.google.com/drive/1lIyfonJgAK1eDe2W-bExaPyUtUt7Nu0t?usp=sharing)

---

### 🔗 Hybrid Model (ResNet50 + EfficientNetB7)
Training notebook for the **Hybrid model combining ResNet50 and EfficientNetB7**:

[Open Hybrid Model Training Notebook](https://colab.research.google.com/drive/1xUYphyOnU4aVrmMq7Jqdf6Ni1mkTWJC2)

---

## ⚠️ Model Generation

To generate the trained model files used in this project:

- Run the **EfficientNetB7 notebook** to obtain  
  `brain_tumor_efficientnet.keras`

- Run the **Hybrid Model notebook** to obtain  
  `Hybrid_Model_Fixed.keras`

After training, place the generated `.keras` files inside the **model/** directory before running the Flask application.

## 📂 Project Structure

```
Brain_Tumor_Detection_Classification/
│
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── model/                     # Folder for trained models
│   ├── Hybrid_Model_Fixed.keras
│   └── brain_tumor_efficientnet.keras
│
├── templates/                 # HTML templates
│   └── index.html
│
├── static/
│   │
│   ├── uploads/               # Uploaded MRI images
│   │   ├── Glioma1.jpg
│   │   ├── Meningioma1.jpg
│   │   ├── pituitary1.jpg
│   │   └── no_tumor1.jpg
│   │
│   └── gradcam/               # Grad-CAM output images
│       ├── Glioma1.jpg
│       ├── Meningioma1.jpg
│       ├── pituitary1.jpg
│       └── no_tumor1.jpg
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VittanalaTejasri/Brain_Tumor_Detection_Classification.git
cd Brain_Tumor_Detection_Classification
```

---

### 2️⃣ Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add Trained Models

Create a **model** folder and place the trained `.keras` files inside it.

```
model/
│
├── Hybrid_Model_Fixed.keras
└── brain_tumor_efficientnet.keras
```

---

### 4️⃣ Run the Application

Start the application using:

```bash
python -m app
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🚀 Features

* Brain tumor classification from MRI images
* Hybrid deep learning architecture
* Web interface for image upload
* Tumor prediction into four categories
* Grad-CAM visualization for model interpretability

---

## 💻 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Flask**
* **HTML / CSS**
* **Convolutional Neural Networks (CNN)**

---

## 👩‍💻 Author

**Batch C11**
- Shri Vishnu Engineering College for Women
