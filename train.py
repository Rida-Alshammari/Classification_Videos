from tensorflow.keras.layers import Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,  # type: ignore
                                     Dense, LSTM, TimeDistributed, Dropout)
from tensorflow.keras.utils import Sequence  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np
import cv2
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.makedirs('./model', exist_ok=True)
os.makedirs('./images', exist_ok=True)

# Todo: Hyperparameters
IMG_SIZE = 64
MAX_FRAMES = 15
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
PREPROCESSED_DIR = 'preprocessed_data'


def preprocess_and_save(video_paths, label):
    """Preprocess and save data in advance"""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    processed_paths = []
    for path in video_paths:
        try:
            # Extract frames from video
            cap = cv2.VideoCapture(path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(total_frames // MAX_FRAMES, 1)

            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = frame / 255.0
                    frames.append(frame)
                if len(frames) >= MAX_FRAMES:
                    break
            cap.release()

            if len(frames) < MAX_FRAMES:
                continue

            # Save frames as numpy array
            base_name = os.path.basename(path).split('.')[0]
            save_path = os.path.join(
                PREPROCESSED_DIR, f"{label}_{base_name}.npy")
            np.save(save_path, frames[:MAX_FRAMES])
            processed_paths.append(save_path)

        except Exception as e:
            print(f"❌ Error Processing: {path}: {str(e)}")

    return processed_paths


class PreprocessedDataGenerator(Sequence):
    """Generator for preprocessed data"""

    def __init__(self, file_paths, labels, batch_size):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx *
                                      self.batch_size: (idx+1)*self.batch_size]
        batch_labels = self.labels[idx *
                                   self.batch_size: (idx+1)*self.batch_size]

        batch_data = []
        for f in batch_files:
            data = np.load(f)
            batch_data.append(data)

        return np.array(batch_data), np.array(batch_labels)


def prepare_dataset():
    """Prepare the dataset and return train and test sets"""
    if not os.path.exists(PREPROCESSED_DIR):
        violent = [os.path.join('dataset/violent', f)
                   for f in os.listdir('dataset/violent')
                   if f.endswith(('.mp4', '.avi', '.mov'))]

        non_violent = [os.path.join('dataset/non_violent', f)
                       for f in os.listdir('dataset/non_violent')
                       if f.endswith(('.mp4', '.avi', '.mov'))]

        print("================================================")
        print("🔃 Preprocessing Violent Videos...")
        violent_paths = preprocess_and_save(violent, 1)
        print("🔃 Preprocessing Non-Violent Videos...")
        non_violent_paths = preprocess_and_save(non_violent, 0)
        print("================================================")

        paths = violent_paths + non_violent_paths
        labels = [1]*len(violent_paths) + [0]*len(non_violent_paths)

    else:
        # تحميل البيانات المُعالجة
        print("🔃 Loading Preprocessed Data...")
        paths = [os.path.join(PREPROCESSED_DIR, f)
                 for f in os.listdir(PREPROCESSED_DIR)
                 if f.endswith('.npy')]

        labels = [int(f.split('_')[0]) for f in os.listdir(PREPROCESSED_DIR)
                  if f.endswith('.npy')]

    return train_test_split(
        paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )


# Todo: Build the model with the new architecture


def build_enhanced_model():
    model = Sequential([
        Input(shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3))),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Flatten()),

        LSTM(128, return_sequences=True, dropout=0.4),
        LSTM(64, dropout=0.4),

        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model

# Todo: Evaluation and visualisation functions


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./images/training_history.png')
    plt.close()


def evaluate_model(model, test_gen):
    y_pred = model.predict(test_gen).round().astype(int)
    y_true = np.concatenate([test_gen[i][1] for i in range(len(test_gen))])

    # Array of confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Violent', 'Violent'],
                yticklabels=['Non-Violent', 'Violent'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('./images/confusion_matrix.png')
    plt.close()

    # Classification Report
    print("\n📃 Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=['Non-Violent', 'Violent']))


def display_sample_predictions(model, test_gen, num_samples=3):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = np.random.randint(0, len(test_gen))
        X_batch, y_batch = test_gen[idx]
        sample_idx = np.random.randint(0, X_batch.shape[0])

        prediction = model.predict(X_batch[sample_idx][np.newaxis, ...])[0][0]
        true_label = y_batch[sample_idx]

        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_batch[sample_idx][0])
        plt.title(
            f'True: {"Violent" if true_label else "Non-Violent"}\nPred: {prediction:.2f}')
        plt.axis('off')
    plt.savefig('./images/sample_predictions.png')
    plt.close()


def get_validation_generator():
    _, X_val, _, y_val = prepare_dataset()
    return PreprocessedDataGenerator(X_val, y_val, BATCH_SIZE)


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = prepare_dataset()

    train_gen = PreprocessedDataGenerator(X_train, y_train, BATCH_SIZE)
    val_gen = PreprocessedDataGenerator(X_val, y_val, BATCH_SIZE)

    model = build_enhanced_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint('./model/best_model.keras', save_best_only=True),
        CSVLogger('training_log.csv'),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save('./model/final_model.keras')
    evaluate_model(model, val_gen)
    plot_training_history(history)
    display_sample_predictions(model, val_gen)
