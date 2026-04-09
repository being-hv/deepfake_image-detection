import os
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from dataset import get_data_generators, prepare_dataset

def evaluate_model(args):
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    print(f"Loading test data from {args.data_dir}")
    # Temporary fallback: creating generators using the root test_dir
    # Since get_data_generators expects the root with train/val/test folders,
    # we can use tf.keras.utils directly if passing the explicit test folder.
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=32,
        image_size=(256, 256),
        shuffle=False
    )
    
    test_ds = prepare_dataset(test_ds, augment=False)
    
    y_true = []
    y_pred_probs = []
    
    print("Running predictions...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate Metrics
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_probs))
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    results["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }
    
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k.capitalize()}: {v:.4f}")
            
    # Save to metrics.json in the model directory
    model_dir = os.path.dirname(args.model_path)
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Model")
    parser.add_argument('--model_path', type=str, default='models/saved_model/best_model.h5', help='Path to compiled h5 model')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Directory to test set')
    
    args = parser.parse_args()
    evaluate_model(args)
