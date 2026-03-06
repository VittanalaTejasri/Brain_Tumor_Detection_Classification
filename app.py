import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# =============================
# FOLDERS
# =============================
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

IMG_SIZE = 224

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# =============================
# LOAD BOTH MODELS
# =============================
classification_model = load_model("model/Hybrid_Model_Fixed (1).keras", compile=False)
gradcam_model = load_model("model/brain_tumor_efficientnet.keras", compile=False)

last_conv_layer_name = "top_conv"


# =============================
# PREPROCESS FOR CLASSIFICATION
# =============================
def preprocess_classification(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =============================
# PREPROCESS FOR GRADCAM
# =============================
def preprocess_gradcam(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# =============================
# GRADCAM FUNCTION
# =============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# =============================
# MAIN ROUTE
# =============================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    uploaded_image = None
    gradcam_image = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # ------------------------
            # MODEL 1: CLASSIFICATION
            # ------------------------
            img_class = preprocess_classification(filepath)
            pred = classification_model.predict(img_class)

            predicted_index = np.argmax(pred)
            prediction = class_names[predicted_index]
            confidence = round(np.max(pred) * 100, 2)

            # ------------------------
            # MODEL 2: GRADCAM
            # ------------------------
            img_grad = preprocess_gradcam(filepath)
            heatmap = make_gradcam_heatmap(img_grad, gradcam_model, last_conv_layer_name)

            original_img = cv2.imread(filepath)
            original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

            heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap = np.uint8(255 * heatmap)
            if prediction == "no_tumor":
                blue_heatmap = np.zeros_like(original_img)
                blue_heatmap[:, :, 0] = 255  # Blue channel
                superimposed_img = cv2.addWeighted(original_img, 0.6, blue_heatmap, 0.4, 0)
            else:
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], file.filename)
            cv2.imwrite(gradcam_path, superimposed_img)

            uploaded_image = filepath
            gradcam_image = gradcam_path
    tumor_info = ""

    if prediction == "glioma":
        tumor_info = "Glioma is a type of tumor that occurs in the brain and spinal cord. It originates in glial cells and can be aggressive depending on the grade."

    elif prediction == "meningioma":
        tumor_info = "Meningioma is a tumor that arises from the meninges, the membranes that surround the brain and spinal cord. It is usually benign and slow-growing."

    elif prediction == "pituitary":
        tumor_info = "Pituitary tumor develops in the pituitary gland and may affect hormone production. Most are non-cancerous."

    elif prediction == "no_tumor":
        tumor_info = "No tumor detected. The MRI scan appears normal without abnormal mass formation."
    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           gradcam_image=gradcam_image,
                             tumor_info=tumor_info)


if __name__ == "__main__":
    app.run(debug=True)