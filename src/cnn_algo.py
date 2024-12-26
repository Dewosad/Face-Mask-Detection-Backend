import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_cnn(data_dir, output_model_path, img_size=128, batch_size=32, epochs=20):
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 80% training, 20% validation split
    )

    # Load images from the subfolders (with_mask, without_mask, incorrect_mask)
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"  # Training subset
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"  # Validation subset
    )

   #CNN architecture
    model = Sequential([
        #covulation layers
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax")  # 3 classes: with_mask, without_mask, mask_wered_incorrect
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    output_dir = os.path.dirname(output_model_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    model.save(output_model_path)
    print(f"Model saved at {output_model_path}")

if name == "main":
    data_directory = "../dataset"  #
    output_model_file = "models/cnn_mask_classifier.keras" 
    train_cnn(data_directory, output_model_file)