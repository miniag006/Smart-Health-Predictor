from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os
import json

from db_sqlite import get_db, init_db
from ml_service import ModelService


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_change_me")
    app.permanent_session_lifetime = timedelta(hours=6)

    
    init_db()
    app.model_service = ModelService(
        base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

    @app.route("/")
    def index():
        if session.get("user_id"):
            return redirect(url_for("home"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")
        data = request.form
        username = data.get("username", "").strip()
        password = data.get("password", "")
        if not username or not password:
            return render_template("login.html", error="Please enter username and password")
        cnx = get_db()
        cur = cnx.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        cur.close()
        if not user or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid username or password")
        session["user_id"] = user["id"]
        session["full_name"] = user["full_name"]
        return redirect(url_for("home"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "GET":
            return render_template("signup.html")
        f = request.form
        required = ["full_name","username","password","age","gender","weight","height","email"]
        if any(not f.get(k) for k in required):
            return render_template("signup.html", error="Please fill all fields")
        cnx = get_db()
        cur = cnx.cursor()
        
        cur.execute("SELECT COUNT(*) FROM users WHERE username=? OR email=?", (f["username"], f["email"]))
        (count,) = cur.fetchone()
        if count:
            cur.close()
            return render_template("signup.html", error="Username or email already exists")
        cur.execute(
            """
            INSERT INTO users(full_name, username, password_hash, age, gender, weight, height, email)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                f["full_name"].strip(),
                f["username"].strip(),
                generate_password_hash(f["password"]),
                int(f["age"]),
                f["gender"],
                float(f["weight"]),
                float(f["height"]),
                f["email"].strip(),
            ),
        )
        cnx.commit()
        cur.close()
        flash("Signup successful. Please login.")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    def dashboard():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return render_template("dashboard.html", full_name=session.get("full_name"))

    @app.route("/home")
    def home():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        first = (session.get("full_name") or "User").split(" ")[0]
        thought = "Health is a state of body. Wellness is a state of being."
      
        stats = {"total": 0, "last": None, "last_date": None, "top": None}
        try:
            cnx = get_db()
            cur = cnx.cursor()
            
            cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (int(session.get("user_id")),))
            (total_count,) = cur.fetchone()
            stats["total"] = int(total_count or 0)
            
            cur.execute(
                "SELECT primary_disease, created_at FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (int(session.get("user_id")),),
            )
            last_row = cur.fetchone()
            if last_row:
                stats["last"] = last_row[0]
                stats["last_date"] = last_row[1]
            
            cur.execute(
                "SELECT primary_disease, COUNT(*) c FROM predictions WHERE user_id=? GROUP BY primary_disease ORDER BY c DESC LIMIT 1",
                (int(session.get("user_id")),),
            )
            top_row = cur.fetchone()
            if top_row:
                stats["top"] = top_row[0]
            cur.close()
        except Exception:
            pass
        return render_template("landing.html", full_name=session.get("full_name"), first_name=first, thought=thought, stats=stats)

    @app.route("/profile")
    def profile():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        cnx = get_db()
        cur = cnx.cursor()
        cur.execute("SELECT full_name, username, age, gender, weight, height, email, created_at FROM users WHERE id=?", (int(session.get("user_id")),))
        row = cur.fetchone()
        cur.close()
        info = None
        if row:
            info = {
                "full_name": row[0],
                "username": row[1],
                "age": row[2],
                "gender": row[3],
                "weight": row[4],
                "height": row[5],
                "email": row[6],
                "created_at": row[7],
            }
        return render_template("profile.html", info=info, full_name=session.get("full_name"))

    @app.route("/predict", methods=["POST"])
    def predict():
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            payload = request.get_json(force=True) or {}
            symptoms = payload.get("symptoms", [])
            result = app.model_service.predict_from_symptoms(symptoms)
            
            try:
                cnx = get_db()
                cur = cnx.cursor()
                cur.execute(
                    """
                    INSERT INTO predictions(user_id, primary_disease, top5_json, symptoms_json)
                    VALUES (?,?,?,?)
                    """,
                    (
                        int(session.get("user_id")),
                        result.get("disease",""),
                        json.dumps(result.get("top5", [])),
                        json.dumps(symptoms),
                    ),
                )
                cnx.commit()
                cur.close()
            except Exception:
                pass
            return jsonify({"ok": True, **result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/features", methods=["GET"])
    def features():
        feats = getattr(app.model_service, "feature_names", []) or []
        return jsonify({"features": feats})

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/history")
    def history():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        cnx = get_db()
        cur = cnx.cursor()
        cur.execute(
            """
            SELECT id, primary_disease, top5_json, symptoms_json, created_at
            FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 100
            """,
            (int(session.get("user_id")),),
        )
        rows = cur.fetchall()
        cur.close()
        
        items = []
        for r in rows:
            try:
                top5 = json.loads(r[2]) if r[2] else []
            except Exception:
                top5 = []
            try:
                syms = json.loads(r[3]) if r[3] else []
            except Exception:
                syms = []
            items.append({
                "id": r[0],
                "primary": r[1],
                "top5": top5,
                "symptoms": syms,
                "created_at": r[4],
            })
        return render_template("history.html", items=items, full_name=session.get("full_name"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
