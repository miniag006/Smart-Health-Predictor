import pandas as pd

class SymptomMapper:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise FileNotFoundError(f"Could not read {csv_file}: {e}")

        # Make sure columns match (case-insensitive handling)
        columns = [c.lower() for c in self.df.columns]
        if "symptom" not in columns or "weight" not in columns:
            raise ValueError("CSV must have 'Symptom' and 'Weight' columns")

        # Normalize column names
        self.df.columns = [c.capitalize() for c in self.df.columns]

        # Create a mapping dictionary
        self.symptom_to_weight = dict(zip(self.df["Symptom"], self.df["Weight"]))

    def list_symptoms(self):
        return list(self.symptom_to_weight.keys())

    def get_weight(self, symptom_name):
        return self.symptom_to_weight.get(symptom_name, 0)


# Quick self-test (runs only if you execute `python symptoms.py`)
if __name__ == "__main__":
    mapper = SymptomMapper("datasets/Symptom-severity.csv")
    print("First 10 symptoms:", mapper.list_symptoms()[:10])
    print("Weight of 'Fever':", mapper.get_weight("Fever"))
