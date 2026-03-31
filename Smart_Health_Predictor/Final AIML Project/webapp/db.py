 import os
 import sqlite3
 
 _cached_cnx = None
 DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")
 
 def get_db():
     global _cached_cnx
     if _cached_cnx is None:
         _cached_cnx = sqlite3.connect(DB_PATH, check_same_thread=False)
         _cached_cnx.row_factory = sqlite3.Row
     return _cached_cnx
 
 
 def init_db():
     cnx = get_db()
     cur = cnx.cursor()
     cur.execute(
         """
         CREATE TABLE IF NOT EXISTS users (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             full_name TEXT NOT NULL,
             username TEXT NOT NULL UNIQUE,
             password_hash TEXT NOT NULL,
             age INTEGER NOT NULL,
             gender TEXT NOT NULL,
             weight REAL NOT NULL,
             height REAL NOT NULL,
             email TEXT NOT NULL UNIQUE,
             created_at TEXT DEFAULT (datetime('now'))
         )
         """
     )
     cnx.commit()
     cur.close()
