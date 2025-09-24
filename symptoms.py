import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

class SymptomMapper:
    # Map string severity labels to numeric weights
    SEVERITY_MAP = {
        "Very Rare": 0.1,
        "Rare": 0.3,
        "Occasional": 0.5,
        "Common": 0.7,
        "Very Common": 1.0
    }

    def __init__(self, csv_path="./datasets/Symptom-severity.csv"):
        """
        Initializes SymptomMapper by reading a CSV containing symptom weights.
        CSV must have columns: 'Symptom', 'weight'
        Weight can be numeric or string (like 'Rare', 'Common', etc.)
        """
        logging.debug("Initializing SymptomMapper...")
        self.symptom_to_weight = {}
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                weight = row['weight']
                # Convert string severity to numeric if needed
                if isinstance(weight, str):
                    weight = weight.strip()
                    weight = self.SEVERITY_MAP.get(weight, 0)  # default 0 if unknown string
                self.symptom_to_weight[row['Symptom']] = float(weight)
            # Store symptom list for reference
            self.symptoms = list(self.symptom_to_weight.keys())
            logging.debug(f"Loaded {len(self.symptoms)} symptoms from {csv_path}")
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
        return self.symptoms[:n]
