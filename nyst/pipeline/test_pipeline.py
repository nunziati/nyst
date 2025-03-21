import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dataset import NystDataset
from classifier import NystClassifier
import sklearn.metrics as skm  # Import necessario per la curva ROC

class TestPipeline:
    def __init__(self, output_dir, test_file_path, std_file_path, model_path, params=None):
        r"""
        params: dict
            params: dictionary containing model parameters: the keys are input_dim, num_channels, nf, index_activation_middle_layer, index_activation_last_layer
            Examples:
            params = {
                "input_dim": 150,
                "num_channels": 8,
                "nf": 32,
                "index_activation_middle_layer": 0,
                "index_activation_last_layer": 0
                }
        """
        self.output_dir = output_dir
        self.test_file_path = test_file_path
        self.std_file_path = std_file_path
        self.model_path = model_path
        self.model = None
        self.test_data = None
        self.std = None
        self.params = params

    def load_model(self):
        if self.params is None:
            self.model = NystClassifier.from_pretrained(self.model_path)
        else:
            self.model = NystClassifier(**self.params)
            self.model.load_weights(self.model_path)

        print(f"Model loaded from {self.model_path}")

    def load_std(self):
        self.std = np.load(self.std_file_path)
        print(f"Target column (std) loaded from {self.std_file_path}")

    def evaluate(self, X, y):
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            predictions = self.model(X_tensor).numpy()
            predictions_binary = (predictions > 0.5).astype(int)  # Assuming binary classification

        accuracy = accuracy_score(y, predictions_binary)
        precision = precision_score(y, predictions_binary, zero_division=0)
        recall = recall_score(y, predictions_binary, zero_division=0)
        f1 = f1_score(y, predictions_binary, zero_division=0)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        return accuracy, precision, recall, f1

    def plot_metrics(self, metrics, confidences):
        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(confidences, metrics['accuracy'], label='Accuracy', color='blue')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence Threshold')
        plt.grid(True)
        output_path = os.path.join(self.output_dir, 'accuracy_plot.png')
        plt.savefig(output_path)
        print(f"Accuracy plot saved to {output_path}")
        plt.close()

        # Plot Precision-Recall
        plt.figure(figsize=(10, 6))
        plt.plot(confidences, metrics['precision'], label='Precision', color='green')
        plt.plot(confidences, metrics['recall'], label='Recall', color='orange')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(self.output_dir, 'precision_recall_plot.png')
        plt.savefig(output_path)
        print(f"Precision-Recall plot saved to {output_path}")
        plt.close()

        # Plot F1 Score
        plt.figure(figsize=(10, 6))
        plt.plot(confidences, metrics['f1_score'], label='F1 Score', color='purple')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Confidence Threshold')
        plt.grid(True)
        output_path = os.path.join(self.output_dir, 'f1_score_plot.png')
        plt.savefig(output_path)
        print(f"F1 Score plot saved to {output_path}")
        plt.close()

    def plot_roc_curve(self, y_true, y_scores):
        # Calculate ROC curve and AUC
        fpr, tpr, _ = skm.roc_curve(y_true, y_scores)
        roc_auc = skm.auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        output_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(output_path)
        print(f"ROC curve plot saved to {output_path}")
        plt.close()

    def run(self):
        self.load_model()
        self.load_std()

        # Load Dataset 
        self.test_data = NystDataset(self.test_file_path, self.std)
        X_test = self.test_data.fil_norm_data
        y_test = self.test_data.extr_data['labels'] 

        confidences = [i / 10 for i in range(1, 10, 0.5)]  # Confidence thresholds from 0.1 to 0.9
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        y_scores = []  # To store raw scores for ROC curve

        for confidence in confidences:
            with torch.no_grad():
                X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                predictions = self.model(X_tensor).numpy()
                y_scores = predictions  # Save raw scores for ROC
                predictions_binary = (predictions > confidence).astype(int)

            metrics['accuracy'].append(accuracy_score(y_test, predictions_binary))
            metrics['precision'].append(precision_score(y_test, predictions_binary, zero_division=0))
            metrics['recall'].append(recall_score(y_test, predictions_binary, zero_division=0))
            metrics['f1_score'].append(f1_score(y_test, predictions_binary, zero_division=0))

        self.plot_metrics(metrics, confidences)
        self.plot_roc_curve(y_test, y_scores)


if __name__ == '__main__':
    # Example usage
    output_dir = '/path/to/output'
    test_file_path = '/path/to/test.csv'
    std_file_path = '/path/to/target.npy'
    model_path = '/path/to/model.pth'

    test_pipeline = TestPipeline(output_dir, test_file_path, std_file_path, model_path)
    test_pipeline.run()