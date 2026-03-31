import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd

class ModelService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.data_dir = os.path.join(base_dir, "data")
        self._load_artifacts()

    def _load_artifacts(self):
        
        self.scaler = self._safe_load(os.path.join(self.models_dir, "scaler.joblib"))
        self.score_scaler = self._safe_load(os.path.join(self.models_dir, "score_scaler.joblib"))
        self.bayesian_models = self._safe_load(os.path.join(self.models_dir, "models_bayesian.joblib"), pickle_load=True)
        self.rf_meta = self._safe_load(os.path.join(self.models_dir, "random_forest_meta.joblib"), pickle_load=True)
        classes_path = os.path.join(self.models_dir, "classes_order.json")
        self.classes = []
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                self.classes = json.load(f)
        
        prepared_train = os.path.join(self.data_dir, "prepared_train.csv")
        if os.path.exists(prepared_train):
            df = pd.read_csv(prepared_train, nrows=1)
            self.feature_names = [c for c in df.columns if c != "prognosis"]
        else:
            self.feature_names = []
        
        info_path = os.path.join(os.path.dirname(__file__), "disease_info.json")
        self.info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    self.info = json.load(f)
            except Exception:
                self.info = {}

    def _safe_load(self, path, pickle_load=False):
        if not os.path.exists(path):
            return None
        try:
            if pickle_load:
                with open(path, "rb") as f:
                    return pickle.load(f)
            return joblib.load(path)
        except Exception:
            return None

    def healthy(self):
        return all([self.scaler, self.score_scaler, self.bayesian_models, self.rf_meta, self.feature_names, self.classes])

    def vectorize_symptoms(self, symptoms):
        
        x = np.zeros((1, len(self.feature_names)), dtype=float)
        symptom_set = set(map(str.lower, symptoms or []))
        for i, feat in enumerate(self.feature_names):
            x[0, i] = 1.0 if feat.lower().replace("_", " ") in symptom_set or feat.lower() in symptom_set else 0.0
        return x

    def predict_from_symptoms(self, symptoms):
        if self.healthy():
            X = self.vectorize_symptoms(symptoms)
            Xn = self.scaler.transform(X)
            
            classes = list(self.bayesian_models.keys())
            scores = np.column_stack([self.bayesian_models[c].predict(Xn) for c in classes])
            scores_n = self.score_scaler.transform(scores)
            pred = self.rf_meta.predict(scores_n)[0]
            disease = str(pred)
            top5 = self._top_k(scores_n, k=5)
            return {
                "disease": disease,
                "description": self._description_for(disease),
                "tips": self._tips_for(disease),
                "top5": top5,
                "more_info_url": self._more_info_for(disease),
            }
       
        return self._fallback(symptoms)

    def _fallback(self, symptoms):
        s = set([x.lower() for x in symptoms or []])
        if {"fever", "cough"} & s and "breath" in " ".join(s):
            d = "Flu-like Illness"
        elif {"itching", "skin_rash"} & s:
            d = "Allergy"
        elif {"headache", "nausea"} & s:
            d = "Migraine"
        else:
            d = "General Checkup Recommended"
        return {
            "disease": d,
            "description": self._description_for(d),
            "tips": self._tips_for(d),
        }

    def _description_for(self, disease):
        
        if disease in self.info and isinstance(self.info[disease], dict):
            return self.info[disease].get("description", "")
        desc = {
            "Migraine": "A neurological condition causing intense headaches often with nausea and sensitivity to light.",
            "Allergy": "Immune response to substances causing itching, sneezing, or rash.",
            "Flu-like Illness": "Viral infection with fever, cough, body aches, and fatigue.",
            "General Checkup Recommended": "Symptoms are non-specific. Please consult a physician for diagnosis.",
        }
        return desc.get(disease, "")

    def _tips_for(self, disease):
        if disease in self.info and isinstance(self.info[disease], dict):
            tips = self.info[disease].get("tips", [])
            if tips:
                return tips
        tips = {
            "Migraine": ["Rest in a dark, quiet room", "Hydrate", "Consult for triptans if frequent"],
            "Allergy": ["Avoid known allergens", "Use antihistamines", "Consult allergist if severe"],
            "Flu-like Illness": ["Hydration and rest", "Paracetamol for fever", "Seek care if breathing difficulty"],
            "General Checkup Recommended": ["Monitor symptoms", "Maintain hydration", "Consult a doctor"],
        }
        return tips.get(disease, [
            "Maintain hydration",
            "Get adequate rest",
            "Consult a healthcare professional if symptoms persist or worsen"
        ])

    def _more_info_for(self, disease):
        if disease in self.info and isinstance(self.info[disease], dict):
            return self.info[disease].get("more_info_url", "")
        return ""

    def _top_k(self, scores_n, k=5):
        if hasattr(self.rf_meta, "predict_proba"):
            proba = self.rf_meta.predict_proba(scores_n)[0]
            cattr = getattr(self.rf_meta, "classes_", None)
            if cattr is None:
                classes = list(self.classes)
            else:
                
                classes = list(cattr.tolist() if hasattr(cattr, "tolist") else cattr)
            pairs = list(zip(classes, proba))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [
                {"disease": str(c), "confidence": float(p)}
                for c, p in pairs[:k]
            ]
        
        return [{"disease": str(self.rf_meta.predict(scores_n)[0]), "confidence": 1.0}]
