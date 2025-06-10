import zipfile
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

zip_path = "/content/Lung_Cancer_Stages_Dataset.zip"  # Update the correct path
extract_path = "/content/dataset/"  # Extract to a folder

# Create the dataset folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully!")
print(os.listdir(extract_path))  # Check extracted files

# Paths
DATASET_DIR = "/content/dataset"
OUTPUT_DIR = "/content/Lung_Cancer_Dataset_Split/"



def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            train_count = int(len(images) * train_ratio)
            val_count = int(len(images) * val_ratio)

            for dataset_type, start, end in [("train", 0, train_count),
                                             ("validation", train_count, train_count + val_count),
                                             ("test", train_count + val_count, len(images))]:
                dataset_path = os.path.join(output_dir, dataset_type, class_name)
                os.makedirs(dataset_path, exist_ok=True)
                for img in images[start:end]:
                    shutil.copy(os.path.join(class_path, img), dataset_path)

# Usage
split_dataset("/content/dataset", "/content/Lung_Cancer_Dataset_Split/")


# Directories
train_dir = "/content/Lung_Cancer_Dataset_Split/train"
val_dir = "/content/Lung_Cancer_Dataset_Split/validation"
test_dir = "/content/Lung_Cancer_Dataset_Split/test"

# Image Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False)

# Input Layer
input_tensor = Input(shape=(224, 224, 3))

# Xception Model
base_xception = Xception(weights="imagenet", include_top=False, input_tensor=input_tensor)
xception_out = GlobalAveragePooling2D()(base_xception.output)

# MobileNetV2 Model
base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
mobilenet_out = GlobalAveragePooling2D()(base_mobilenet.output)

# Merge Features
merged = Concatenate()([xception_out, mobilenet_out])
merged = Dense(512, activation="relu")(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.5)(merged)

merged = Dense(256, activation="relu")(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.4)(merged)

merged = Dense(128, activation="relu")(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.3)(merged)

# Output Layer
output = Dense(5, activation="softmax")(merged)

# Final Model
model = Model(inputs=input_tensor, outputs=output)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=25)

# Save Model
model.save("lung_cancer_model.h5")

# Evaluate on Test Data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Get true labels and predictions
true_labels = test_generator.classes  # Assuming test_generator is your test data
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Generate Classification Report
class_names = list(test_generator.class_indices.keys())
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", report)

# Generate Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



# Function to preprocess and predict a single image
def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.show()

    return class_names[predicted_class]

# Test on Sample Images
sample_images = ["/content/Lung_Cancer_Dataset_Split/test/Bengin cases/Bengin case (111).jpg", "/content/Lung_Cancer_Dataset_Split/test/Normal cases/Normal case (127).jpg",
                 "/content/Lung_Cancer_Dataset_Split/test/Stage_1/Malignant case (180).jpg",
                 "/content/Lung_Cancer_Dataset_Split/test/Stage_2/Malignant case (116).jpg",
                 "/content/Lung_Cancer_Dataset_Split/test/Stage_3/Malignant case (447).jpg"]  # Replace with actual image paths
class_names = list(test_generator.class_indices.keys())  # Get class labels

for img_path in sample_images:
    print(f"Testing: {img_path}")
    predicted_class = predict_image(img_path, model, class_names)
    print(f"Prediction: {predicted_class}")


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = np.dot(conv_outputs, pooled_grads.numpy())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_gradcam(image_path, model, last_conv_layer_name, alpha=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

# Example Usage:
overlay_gradcam("/content/Lung_Cancer_Dataset_Split/test/Stage_3/Malignant case (447).jpg", model, "block14_sepconv2")
