import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils.dataset_handler import get_datasets
from utils.model_builder import build_all_models

# Configure GPU memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DATA_DIR = "dataset/"
MODELS_DIR = "models/"
TRAINING_DIR = "training/"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(os.path.join(TRAINING_DIR, f"{model_name}_history.png"))
    plt.close()

def plot_confusion_matrix(model, test_ds, class_names, model_name):
    print(f"Evaluating {model_name} for Confusion Matrix...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(TRAINING_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Save class names for prediction script
    with open(os.path.join(MODELS_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(class_names))

    print("Building models...")
    models_dict = build_all_models(num_classes)

    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Training Model: {name}")
        print(f"{'='*50}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save extension, .keras is safer for sub-classed HF models
        model_path = os.path.join(MODELS_DIR, f"{name}.keras")
        
        if os.path.exists(model_path):
            print(f"Model {name} already exists at {model_path}. Skipping training to save time.")
            continue
            
        # Unfreeze top layers to hit 97-98% accuracy in just 5 Epochs
        try:
            inner_base = model.layers[1] if len(model.layers) > 1 else model
            inner_base.trainable = True
            for layer in inner_base.layers[:-20]: # Keep most frozen, unfreeze top 20
                layer.trainable = False
        except:
            pass
            
        # Re-compile model since we changed trainability
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,
            callbacks=callbacks
        )
        
        # Plot and save
        plot_history(history, name)
        
        # Evaluate Best model (Since it was restored via EarlyStopping or ModelCheckpoint)
        eval_model = models_dict[name] 
        plot_confusion_matrix(eval_model, test_ds, class_names, name)

if __name__ == "__main__":
    main()
