import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

class TestPipeline:
    def __init__(self, root_dir, test_file, model_path, std):
        self.root_dir = root_dir
        self.test_file = test_file
        self.model_path = model_path
        self.std = std
        self.model = None
        self.test_data = None

    def load_model(self):
        model_full_path = os.path.join(self.root_dir, self.model_path)
        self.model = joblib.load(model_full_path)
        print(f"Model loaded from {model_full_path}")

    def load_test_data(self):
        test_file_full_path = os.path.join(self.root_dir, self.test_file)
        self.test_data = pd.read_csv(test_file_full_path)
        print(f"Test data loaded from {test_file_full_path}")

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Model accuracy: {accuracy}")
        return accuracy

    def run(self):
        self.load_model()
        self.load_test_data()
        
        X_test = self.test_data.drop(columns=[self.std])
        y_test = self.test_data[self.std]
        
        self.evaluate(X_test, y_test)


if __name__ == '__main__':
    # Example usage
    root_dir = '/path/to/root'
    test_file = 'data/test.csv'
    model_path = 'models/model.joblib'
    std = 'target_column'
    
    test_pipeline = TestPipeline(root_dir, test_file, model_path, std)
    test_pipeline.run()