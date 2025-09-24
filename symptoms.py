import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

class SymptomMapper:
    def __init__(self, csv_path="./datasets/Symptom-severity.csv"):
        """
        Initializes SymptomMapper by reading a CSV containing symptom weights.
        CSV must have columns: 'Symptom', 'weight'
        """
        logging.debug("Initializing SymptomMapper...")
        self.symptom_to_weight = {}
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.symptom_to_weight[row['Symptom']] = row['weight']
            logging.debug(f"Loaded {len(self.symptom_to_weight)} symptoms from {csv_path}")
        except Exception as e:
            logging.error(f"Failed to load symptoms from {csv_path}: {e}")
    
    def get_weight(self, symptom_name):
        """
        Returns the weight of the given symptom, or 0 if not found.
        """
        return self.symptom_to_weight.get(symptom_name, 0)
    
    def list_symptoms(self, n=10):
        """
        Returns a list of first n symptoms.
        """
        return list(self.symptom_to_weight.keys())[:n]
