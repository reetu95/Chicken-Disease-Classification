import sys
import os

# Clear out any previous paths in the notebook by restarting kernel

# Append the correct path to the 'src' directory inside the nested folder
sys.path.append('/Users/reetu/Documents/Personal_Projects/chicken_disease_classification/Chicken-Disease-Classification/src')

from cnnClassifier.entity import TrainingConfig
import tensorflow as tf 
from pathlib import Path


from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Instead of loading an old model with batch_shape, we rebuild it using input_shape
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,  # Use input_shape to avoid the batch_shape issue
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Add custom layers if needed (e.g., for classification tasks)
        flatten_in = tf.keras.layers.Flatten()(self.model.output)
        prediction = tf.keras.layers.Dense(
            units=self.config.params_classes,
            activation="softmax"
        )(flatten_in)

        # Build the full model
        self.model = tf.keras.models.Model(inputs=self.model.input, outputs=prediction)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save the newly built model
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    def manual_image_loader(self, image_dir, label_map):
        images = []
        labels = []
        
        for label, idx in label_map.items():
            class_dir = os.path.join(image_dir, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path)
                    img = img.resize(self.config.params_image_size[:-1])  # Resize image to match model input
                    img = img_to_array(img)
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue

        images = np.array(images) / 255.0  # Normalize the pixel values between 0 and 1
        labels = to_categorical(labels, num_classes=self.config.params_classes)
        
        return images, labels

    def train_valid_generator(self):
        # Define a label map for your dataset (e.g., {"healthy": 0, "coccidiosis": 1})
        label_map = {"healthy": 0, "coccidiosis": 1}

        # Manually load the images and labels
        all_images, all_labels = self.manual_image_loader(self.config.training_data, label_map)

        # Split into training and validation sets
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(
            all_images, all_labels, test_size=0.20, random_state=42)

    def save_model(self, path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        # Fit the model manually using the loaded data
        self.model.fit(
            self.train_images, self.train_labels,
            validation_data=(self.valid_images, self.valid_labels),
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            callbacks=callback_list
        )

        # Save the model after training
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
