import numpy as np
import tensorflow as tf
import time
import psutil
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Concatenate
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Dropout, LSTM, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Attention, GlobalAveragePooling1D
# Step 1: Data Preparation
def prepare_data():

    df = pd.read_pickle('/Users/athenasaghi/VSProjects/CognitiveFatigueDetection/processed_data_full.pkl')
    X_raweeg = np.stack(df['raweeg'].values)
    X_features = np.stack(df['features'].values)
    Y_labels = df['label'].values
    # Remove class 3
    mask = Y_labels != 3
    X_raweeg = X_raweeg[mask]
    X_features = X_features[mask]
    Y_labels = Y_labels[mask]

    num_classes = len(np.unique(Y_labels))
    Y_labels = to_categorical(Y_labels, num_classes=num_classes)

    scaler = StandardScaler()
    X_raweeg = scaler.fit_transform(X_raweeg.reshape(-1, X_raweeg.shape[-1])).reshape(X_raweeg.shape)
    X_features = scaler.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)

    # Replace padding zeros with small noise to avoid learning issues
    X_raweeg[X_raweeg == 0] = np.random.normal(loc=0, scale=1e-6, size=np.sum(X_raweeg == 0))

    # Oversample minority classes
    X_resampled, Y_resampled, Xf_resampled = [], [], []
    labels = np.argmax(Y_labels, axis=1)
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        X_cls_raweeg = X_raweeg[cls_indices]
        X_cls_features = X_features[cls_indices]
        Y_cls = Y_labels[cls_indices]
        X_cls_raweeg, Y_cls, X_cls_features = resample(X_cls_raweeg, Y_cls, X_cls_features, n_samples=max([len(np.where(labels == c)[0]) for c in np.unique(labels)]), random_state=42)
        X_resampled.append(X_cls_raweeg)
        Y_resampled.append(Y_cls)
        Xf_resampled.append(X_cls_features)

    X_raweeg = np.vstack(X_resampled)
    Y_labels = np.vstack(Y_resampled)
    X_features = np.vstack(Xf_resampled)

    # Split into train and test sets
    X_train_raweeg, X_test_raweeg, X_train_features, X_test_features, Y_train, Y_test = train_test_split(
        X_raweeg, X_features, Y_labels, test_size=0.2, random_state=42, stratify=np.argmax(Y_labels, axis=1)
    )

    scaler_raweeg = MinMaxScaler()
    scaler_features = MinMaxScaler()

    X_train_raweeg = scaler_raweeg.fit_transform(X_train_raweeg.reshape(-1, X_train_raweeg.shape[-1])).reshape(X_train_raweeg.shape)
    X_test_raweeg = scaler_raweeg.transform(X_test_raweeg.reshape(-1, X_test_raweeg.shape[-1])).reshape(X_test_raweeg.shape)

    X_train_features = scaler_features.fit_transform(X_train_features.reshape(-1, X_train_features.shape[-1])).reshape(X_train_features.shape)
    X_test_features = scaler_features.transform(X_test_features.reshape(-1, X_test_features.shape[-1])).reshape(X_test_features.shape)

    X_train_raweeg += 0.005 * np.random.randn(*X_train_raweeg.shape)
    X_train_features += 0.01 * np.random.randn(*X_train_features.shape)

    return X_train_raweeg, X_test_raweeg, X_train_features, X_test_features, Y_train, Y_test

# Step 2: Create Encoders
def create_cnn_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation=None)(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# Step 3: Contrastive Training
def contrastive_training(X_train_raweeg, X_train_features, encoder_raweeg, encoder_features, epochs=400, batch_size=64,wight_decay=1e-3):
    def contrastive_loss(z_raweeg, z_features, temperature=0.2):
        epsilon = 1e-6
        z_raweeg = tf.math.l2_normalize(z_raweeg + epsilon, axis=1)
        z_features = tf.math.l2_normalize(z_features + epsilon, axis=1)
        logits = tf.matmul(z_raweeg, z_features, transpose_b=True) / temperature
        labels = tf.range(tf.shape(logits)[0])
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss)

    lr_schedule = ExponentialDecay(1e-4, 1000, 0.96, staircase=True)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wight_decay, clipnorm=0.6)

    start_time = time.time()

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train_raweeg))
        X_raweeg_shuffled, X_features_shuffled = X_train_raweeg[idx], X_train_features[idx]
        losses = []

        for i in range(0, len(X_train_raweeg) - batch_size + 1, batch_size):
            x_r_batch = X_raweeg_shuffled[i:i + batch_size]
            x_f_batch = X_features_shuffled[i:i + batch_size]

            with tf.GradientTape() as tape:
                z_r = encoder_raweeg(x_r_batch, training=True)
                z_f = encoder_features(x_f_batch, training=True)
                loss = contrastive_loss(z_r, z_f)

            grads = tape.gradient(loss, encoder_raweeg.trainable_variables + encoder_features.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder_raweeg.trainable_variables + encoder_features.trainable_variables))
            losses.append(loss.numpy())

        print(f"Epoch {epoch + 1}/{epochs}, Contrastive Loss: {np.mean(losses):.4f}")

    end_time = time.time()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1e6
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in encoder_raweeg.trainable_variables + encoder_features.trainable_variables])

    print(f"Contrastive Training - Time: {end_time - start_time:.2f}s, RAM Usage: {ram_usage:.2f}MB, Trainable Params: {trainable_params}")

# Step 4: Classification Training
def classification_training(X_train_raweeg, X_train_features, Y_train, encoder_raweeg, encoder_features, num_classes, epoches, lr, weight_decay, k):

 
    # Stratified K-Folds Cross-Validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    true_labels, pred_labels = [], []

    start_time = time.time()

    metrics_all = { 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'auc': [], 'val_auc': [],
                'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [],
                'f1': [], 'val_f1': [], 'specificity': [], 'val_specificity': [], 'confusion_matrix': []}

    metrics_per_class = { f'{metric}_class_{i}': [] for i in range(num_classes)
                      for metric in ['precision', 'recall', 'f1', 'specificity',
                                     'val_precision', 'val_recall', 'val_f1', 'val_specificity','val_acc'] }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raweeg, np.argmax(Y_train, axis=1))):
        X_train_raweeg_fold, X_val_raweeg_fold = X_train_raweeg[train_idx], X_train_raweeg[val_idx]
        X_train_features_fold, X_val_features_fold = X_train_features[train_idx], X_train_features[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

        input_raweeg = Input(shape=(X_train_raweeg.shape[1], X_train_raweeg.shape[2]))
        input_features = Input(shape=(X_train_features.shape[1], X_train_features.shape[2]))
        z_raweeg, z_features = encoder_raweeg(input_raweeg), encoder_features(input_features)
        combined = Concatenate()([z_raweeg, z_features])
        output = Dense(num_classes, activation='softmax')(combined)
        classifier = Model(inputs=[input_raweeg, input_features], outputs=output)

        classifier.compile(optimizer=AdamW(learning_rate=lr, weight_decay=weight_decay),
                       loss='categorical_crossentropy',
                       metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.SpecificityAtSensitivity(0.5, name='specificity')] +
                                [tf.keras.metrics.Precision(name=f'precision_class_{i}', class_id=i) for i in range(num_classes)] +
                                [tf.keras.metrics.Recall(name=f'recall_class_{i}', class_id=i) for i in range(num_classes)])


        classifier.fit([X_train_raweeg_fold, X_train_features_fold], Y_train_fold,
                       validation_data=([X_val_raweeg_fold, X_val_features_fold], Y_val_fold),
                       epochs=epochs, batch_size=128, verbose=1,
                       callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)])

    end_time = time.time()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1e6
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in classifier.trainable_variables])

    print(f"Classification Training - Time: {end_time - start_time:.2f}s, RAM Usage: {ram_usage:.2f}MB, Trainable Params: {trainable_params}")

    return classifier

# Step 5: Evaluate on Test Data
def evaluate_test(classifier, X_test_raweeg, X_test_features, Y_test):
    y_pred_test = classifier.predict([X_test_raweeg, X_test_features])
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    y_true_classes_test = np.argmax(Y_test, axis=1)

    test_accuracy = accuracy_score(y_true_classes_test, y_pred_classes_test)
    test_auc = roc_auc_score(Y_test, y_pred_test, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true_classes_test, y_pred_classes_test)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

# Main Function
def main():

    params = {
        'epochs': [400,450,500],  # Number of epochs for classification training
        'k_folds': [5,8,10],
        'learning_rate': [1e-4,1e-3],  # Learning rate for the optimizer
        'weight_decay': [1e-4,1e-05]  # Weight decay for the optimizer
    }

    for e in params['epochs']:
        for k in params['k_folds']:
            for lr in params['learning_rate']:
                for wd in params['weight_decay']:
                    X_train_raweeg, X_test_raweeg, X_train_features, X_test_features, Y_train, Y_test = prepare_data()

                    encoder_raweeg = create_cnn_encoder(X_train_raweeg.shape[1:])
                    encoder_features = create_cnn_encoder(X_train_features.shape[1:])

                    contrastive_training(X_train_raweeg, X_train_features, encoder_raweeg, encoder_features, wight_decay=wd,epoches =e)
                    classifier = classification_training(X_train_raweeg, X_train_features, Y_train, encoder_raweeg, encoder_features, Y_train.shape[1],k=k, epoches=e, lr=lr, weight_decay=wd)
                    evaluate_test(classifier, X_test_raweeg, X_test_features, Y_test)

if __name__ == "__main__":
    main()