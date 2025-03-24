import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from nyst.dataset import NystDataset
from nyst.classifier import NystClassifier
import sklearn.metrics as skm  # Import necessario per la curva ROC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Output directory created at {self.output_dir}")

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
    
    def save_auc_report(self, y_true, y_scores):
        # Calcola l'AUC
        fpr, tpr, _ = skm.roc_curve(y_true, y_scores)
        roc_auc = skm.auc(fpr, tpr)

        # Salva l'AUC in un file di report
        report_path = os.path.join(self.output_dir, 'report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(f"Area Under the Curve (AUC): {roc_auc:.4f}\n")
        print(f"AUC report saved to {report_path}")

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

    
    def plot_confusion_matrix(self, y_true, y_pred, confidence_threshold):
        """
        Plots and saves the confusion matrix.
        """
        # Calcola la confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

        # Plot della confusion matrix
        plt.figure(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix (Threshold = {confidence_threshold})')
        plt.grid(False)

        # Salva il grafico
        output_path = os.path.join(self.output_dir, f'confusion_matrix_{confidence_threshold:.2f}.png')
        plt.savefig(output_path)
        print(f"Confusion matrix plot saved to {output_path}")
        plt.close()

    def run(self):
        self.load_model()
        self.load_std()

        # Load Dataset 
        self.test_data = NystDataset(self.test_file_path, self.std)
        X_test = self.test_data.fil_norm_data
        y_test = self.test_data.extr_data['labels'] 

        confidences = np.arange(0., 1., 0.05)  # Confidence thresholds from 0. to 1.
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        y_scores = []  # To store raw scores for ROC curve

        for confidence in confidences:
            with torch.no_grad():
                predictions = self.model(X_test).numpy()
                y_scores = predictions  # Save raw scores for ROC
                predictions_binary = (predictions > confidence).astype(int)

            metrics['accuracy'].append(accuracy_score(y_test, predictions_binary))
            metrics['precision'].append(precision_score(y_test, predictions_binary, zero_division=0))
            metrics['recall'].append(recall_score(y_test, predictions_binary, zero_division=0))
            metrics['f1_score'].append(f1_score(y_test, predictions_binary, zero_division=0))

            # Plot confusion matrix for a specific threshold (e.g., 0.5)
            if confidence == 0.5:
                self.plot_confusion_matrix(y_test, predictions_binary, confidence)


        self.plot_metrics(metrics, confidences)
        self.plot_roc_curve(y_test, y_scores)

        # Salva il report AUC
        self.save_auc_report(y_test, y_scores)


if __name__ == '__main__':
    # Define paths
    input_dir = '/repo/porri/nyst/models'  # Directory containing subfolders with models
    output_dir = '/repo/porri/nyst_labelled_videos/grafici'
    test_file_path = '/repo/porri/nyst_labelled_videos/test_dataset.csv'
    std_file_path = '/repo/porri/nyst_labelled_videos/std.npy'

    # Parameters for the model
    params = {
        "input_dim": 150,
        "num_channels": 8,
        "nf": 8,
        "index_activation_middle_layer": 0,
        "index_activation_last_layer": 0
    }

    # Iterate over all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        model_dir = os.path.join(input_dir, subdir)
        if os.path.isdir(model_dir):  # Check if it's a directory
            model_path = os.path.join(model_dir, 'best_model.pth')
            if os.path.exists(model_path):  # Check if the model file exists
                # Create a corresponding output subdirectory
                sub_output_dir = os.path.join(output_dir, subdir)
                os.makedirs(sub_output_dir, exist_ok=True)

                print(f"Processing model in {model_dir}...")

                # Run the pipeline for the current model
                test_pipeline = TestPipeline(sub_output_dir, test_file_path, std_file_path, model_path, params=params)
                test_pipeline.run()
            else:
                print(f"Model file not found in {model_dir}, skipping...")