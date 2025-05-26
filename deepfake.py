# deepfake_detection.py

import os
import glob
import shutil
import zipfile
import random
import gdown
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam


# Step 1: Download and Extract Dataset
def download_and_extract_dataset():
    file_id = "1avWy7Au-BqIzsu3gtmvO0fyQAqk14_SN"
    output = "deepfake_dataset.zip"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("deepfake_dataset")


# Step 2: Organize Dataset into Train/Val/Test
def organize_dataset():
    src_base = "deepfake_dataset/real_and_fake_face_detection/real_and_fake_face"
    for split in ["train", "val", "test"]:
        os.makedirs(f"processed_dataset/{split}/real", exist_ok=True)
        os.makedirs(f"processed_dataset/{split}/fake", exist_ok=True)

    def split_and_move(class_name, label_folder):
        files = glob.glob(os.path.join(src_base, label_folder, "*.jpg"))
        print(f"Found {len(files)} files in {os.path.join(src_base, label_folder)}")

        if len(files) == 0:
            print(f"No files found for {class_name} in {label_folder}. Check the folder path and contents.")
            return

        train, temp = train_test_split(files, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        for split, data in zip(['train', 'val', 'test'], [train, val, test]):
            for f in data:
                dest = f"processed_dataset/{split}/{class_name}/" + os.path.basename(f)
                shutil.copy(f, dest)

    split_and_move("real", "training_real")
    split_and_move("fake", "training_fake")


# Step 3: Create Data Generators
def create_data_generators():
    base_dir = 'processed_dataset'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255)
    img_size = (224, 224)
    batch_size = 128

    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, class_mode='binary', batch_size=batch_size)
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, class_mode='binary', batch_size=batch_size)
    test_gen = datagen.flow_from_directory(test_dir, target_size=img_size, class_mode='binary', batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen


# Step 4: Build and Train Models
def build_and_train_models(train_gen, val_gen, test_gen):
    input_tensor = Input(shape=(224, 224, 3))

    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for model in [vgg, resnet, inception]:
        model.trainable = False

    vgg_feat = GlobalAveragePooling2D()(vgg.output)
    resnet_feat = GlobalAveragePooling2D()(resnet.output)
    inception_feat = GlobalAveragePooling2D()(inception.output)

    merged = Concatenate()([vgg_feat, resnet_feat, inception_feat])
    dense = Dense(256, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense)

    ensemble_model = Model(inputs=input_tensor, outputs=output)

    vgg_model = Model(inputs=input_tensor, outputs=vgg_feat)
    resnet_model = Model(inputs=input_tensor, outputs=resnet_feat)
    inception_model = Model(inputs=input_tensor, outputs=inception_feat)

    for model in [vgg_model, resnet_model, inception_model]:
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=1)
        test_loss, test_acc = model.evaluate(test_gen)
        print(f'Model: {model.name}, Test accuracy: {test_acc:.2f}')
        model.save(f'{model.name}.h5')

    # Fine-tuning
    for layer in vgg.layers[-5:]:
        layer.trainable = True
    for layer in resnet.layers[-5:]:
        layer.trainable = True
    for layer in inception.layers[-5:]:
        layer.trainable = True

    ensemble_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    ensemble_model.fit(train_gen, validation_data=val_gen, epochs=5)
    ensemble_model.save("ensemble_model.h5")


# Run steps
if __name__ == "__main__":
    download_and_extract_dataset()
    organize_dataset()
    train_gen, val_gen, test_gen = create_data_generators()
    build_and_train_models(train_gen, val_gen, test_gen)
