# HealthAI Webapp

Run backend (SQLite, no external DB required):

1. Set optional env var:
   - FLASK_SECRET_KEY
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare and train (from project root):
   ```bash
   python preprocess.py --train ../Training.csv --test ../Testing.csv --out ../data
   python train.py --train data/prepared_train.csv --test data/prepared_test.csv --out models
   ```
4. Run app:
   ```bash
   python app.py
   ```

ML artifacts expected in ../models and features from ../data/prepared_train.csv. SQLite DB file stored at webapp/app.db.
