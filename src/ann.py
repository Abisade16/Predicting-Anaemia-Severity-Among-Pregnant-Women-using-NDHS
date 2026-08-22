#ann.py

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import create_scaled_preprocessor

SEED = 7

tf.random.set_seed(SEED)
np.random.seed(SEED)

def build_ann(input_dim):
    """
    Build the Artificial Neural Network architecture.
    """

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.1),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_ann(X_train, y_train, random_state=42):
    """
    Train the ANN using balanced class weights.

    Returns
    -------
    model : tensorflow.keras.Model
        Trained ANN.
    preprocessor : transformer
        Fitted preprocessing transformer.
    history : History
        Keras training history.
    """
    
    X_train_ann, X_val_ann, y_train_ann, y_val_ann = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    preprocessor = create_scaled_preprocessor()

    X_train_prep = preprocessor.fit_transform(
        X_train_ann,
        y_train_ann
    )

    X_val_prep = preprocessor.transform(
        X_val_ann
    )

    model = build_ann(
        input_dim=X_train_prep.shape[1]
    )

    classes = np.unique(y_train_ann)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_ann
    )

    class_weights = dict(
        zip(classes, weights)
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_prep,
        y_train_ann,
        validation_data=(X_val_prep, y_val_ann),
        epochs=50,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    return model, preprocessor, history

def train_ann_smote(X_train, y_train, random_state=42):
    """
    Train ANN using SMOTE to address class imbalance.

    SMOTE is applied only to the training portion.
    """

    X_train_ann, X_val_ann, y_train_ann, y_val_ann = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    preprocessor = create_scaled_preprocessor()

    X_train_prep = preprocessor.fit_transform(
        X_train_ann,
        y_train_ann
    )

    X_val_prep = preprocessor.transform(
        X_val_ann
    )

    smote = SMOTE(
        random_state=random_state
    )

    X_train_sm, y_train_sm = smote.fit_resample(
        X_train_prep,
        y_train_ann
    )

    model = build_ann(
        input_dim=X_train_sm.shape[1]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_sm,
        y_train_sm,
        validation_data=(X_val_prep, y_val_ann),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    return model, preprocessor, history

def predict_ann(model, preprocessor, X):
    """
    Generate class predictions and probabilities
    from a trained ANN.
    """

    X_prep = preprocessor.transform(X)

    probabilities = model.predict(
        X_prep,
        verbose=0
    )

    predictions = probabilities.argmax(axis=1)

    return predictions, probabilities