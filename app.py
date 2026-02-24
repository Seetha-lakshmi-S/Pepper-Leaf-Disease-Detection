import os
import json
import gc
import re
import pymysql
import cloudinary
import cloudinary.uploader
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from markupsafe import Markup
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.pool import NullPool

# Import your model prediction logic
from predict_utils import run_prediction 

# Required for local MySQL
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# --- 1. DUAL-DATABASE CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # RENDER/SUPABASE SETUP
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    # Use NullPool for Supabase Transaction Mode (Port 6543)
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "poolclass": NullPool,
        "connect_args": {"sslmode": "require"}
    }
    print("Running on RENDER with Supabase (PostgreSQL)")
else:
    # LOCAL MYSQL SETUP
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Seetha%40123@localhost/pepguard_db'
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "pool_pre_ping": True,
        "pool_recycle": 280
    }
    print("Running LOCALLY with MySQL")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cloudinary Setup
cloudinary.config( 
    cloud_name = "do7xycqmw", 
    api_key = "575413169736574", 
    api_secret = "eCJJemDOYQ17J8jLVcKqS8WTVLo",
    secure = True
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- 2. HELPERS (Clean text, Timezones, etc.) ---
def strip_citations(text):
    if not text: return ""
    text = re.sub(r'\[cite_start\]|\[cite_end\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:cite|source):\s*\d+.*?\]', '', text, flags=re.IGNORECASE)
    return text.strip()

def get_ist_time(dt=None):
    if dt is None: dt = datetime.now()
    return dt.astimezone(ZoneInfo("Asia/Kolkata"))

@app.template_filter('clean_text')
def clean_text_filter(s): return strip_citations(s)

@app.template_filter('ist')
def ist_time_filter(dt):
    if dt is None: return ""
    return get_ist_time(dt).strftime('%d %b %Y, %I:%M %p')

@app.template_filter('nl2br')
def nl2br(value):
    if not value: return ""
    return Markup("<br>".join(str(value).splitlines()))

# --- 3. MODELS ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Prediction(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plant_id = db.Column(db.String(150), nullable=False)
    image_filename = db.Column(db.String(500), nullable=False) 
    result = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: get_ist_time())

class DiseaseInfo(db.Model):
    __tablename__ = 'disease_info'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    organic_treatment = db.Column(db.Text)
    chemical_treatment = db.Column(db.Text)
    yield_advice = db.Column(db.Text)
    follow_up = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id): return db.session.get(User, int(user_id))

# --- 4. INITIALIZATION ---
def seed_database_from_json():
    try:
        json_path = os.path.join(app.root_path, 'disease_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if not DiseaseInfo.query.filter_by(name=item['name']).first():
                    db.session.add(DiseaseInfo(**item))
            db.session.commit()
            print("Database synchronized.")
    except Exception as e: print(f"Seed Error: {e}")

@app.before_request
def create_tables():
    app.before_request_funcs[None].remove(create_tables)
    db.create_all()
    seed_database_from_json()

# --- 5. ROUTES ---

@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    local_path = None
    try:
        f = request.files.get('leaf_image')
        if not f: return jsonify({'success': False, 'message': 'No file'})
        
        # Save temp file for prediction
        filename = secure_filename(f"{datetime.now().timestamp()}_{f.filename}")
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(local_path)
        
        # AI Logic
        disease, stage, conf = run_prediction(local_path)
        
        # --- HYBRID STORAGE LOGIC ---
        if DATABASE_URL and "pooler.supabase.com" in DATABASE_URL:
            # LIVE (Render): Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(local_path, folder="pepguard_uploads")
            final_db_path = upload_result['secure_url']
            # Delete local file to save space on Render
            if os.path.exists(local_path): os.remove(local_path)
        else:
            # LOCAL (Your Computer): Keep in static/uploads
            final_db_path = f"uploads/{filename}"

        new_pred = Prediction(
            user_id=current_user.id, 
            plant_id=request.form.get('plant_id', 'Unknown'), 
            image_filename=final_db_path, 
            result=disease, 
            severity=stage, 
            confidence=conf
        )
        db.session.add(new_pred)
        db.session.commit()
        return jsonify({'success': True, 'redirect': url_for('show_result', prediction_id=new_pred.id)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        gc.collect()

@app.route('/')
def landing():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('layout.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash('Username exists', 'error')
            return redirect(url_for('login'))
        hashed_pw = generate_password_hash(request.form.get('password'), method='pbkdf2:sha256')
        db.session.add(User(username=username, password=hashed_pw))
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/get_panel/<path:panel_name>')
@login_required
def get_panel(panel_name):
    if panel_name == 'welcome': return render_template('panels/welcome.html')
    elif panel_name == 'new_diagnosis': return render_template('panels/new_diagnosis.html')
    elif panel_name == 'history':
        base = Prediction.query.filter_by(user_id=current_user.id).all()
        history_data = {}
        for p in sorted(base, key=lambda x: x.timestamp, reverse=True):
            if p.plant_id not in history_data: history_data[p.plant_id] = {'entries': []}
            history_data[p.plant_id]['entries'].append(p)
        return render_template('panels/history.html', history_data=history_data)
    elif panel_name.startswith('result/'):
        pred_id = int(panel_name.split('/')[-1])
        pred = db.session.get(Prediction, pred_id)
        info = DiseaseInfo.query.filter_by(name=pred.severity).first() or \
               DiseaseInfo.query.filter_by(name=pred.result).first()
        return render_template('panels/result.html', prediction=pred, info=info)
    return "Panel not found", 404

@app.route('/result/<int:prediction_id>')
@login_required
def show_result(prediction_id):
    return redirect(url_for('dashboard') + f'#result-{prediction_id}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)