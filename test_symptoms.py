from symptoms import SymptomMapper

print("DEBUG: test_symptoms.py started")
mapper = SymptomMapper("datasets/Symptom-severity.csv")
print("DEBUG: SymptomMapper initialized")
print("First 10 symptoms:", mapper.list_symptoms()[:10])
print("Weight of 'Fever':", mapper.get_weight("Fever"))
