import os
import json
import pymysql
from datetime import datetime
from zoneinfo import ZoneInfo

# --- 1. MEMORY MANAGEMENT (CRITICAL FOR RENDER FREE TIER) ---
# These MUST come before importing tensorflow or your models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress internal TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU only (Render has no GPU)

import tensorflow as tf
# Limit memory usage for the CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.backend.clear_session()

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from markupsafe import Markup

import cloudinary
import cloudinary.uploader

# Import your prediction utility
from predict_utils import run_prediction 

# Allow pymysql to act as MySQLdb
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# --- 2. CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# Database Setup: Prioritize Render's PostgreSQL or fallback to local
db_url = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:Seetha%40123@localhost/pepguard_db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

# --- 3. MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plant_id = db.Column(db.String(150), nullable=False)
    image_filename = db.Column(db.String(500), nullable=False) 
    result = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("Asia/Kolkata")))

class DiseaseInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    organic_treatment = db.Column(db.Text)
    chemical_treatment = db.Column(db.Text)
    yield_advice = db.Column(db.Text)
    follow_up = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 4. ROUTES ---
@app.route('/')
def landing():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('layout.html')

@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    try:
        plant_id = request.form.get('plant_id')
        f = request.files.get('leaf_image')
        if not f: return jsonify({'success': False, 'message': 'No file'})
        
        # Save locally for prediction
        filename = secure_filename(f"{datetime.now().timestamp()}_{f.filename}")
        local_path = filename
        f.save(local_path)
        
        # 1. Cloudinary Upload
        upload_result = cloudinary.uploader.upload(local_path, folder="pepguard_uploads")
        permanent_url = upload_result['secure_url']
        
        # 2. AI Prediction (Heavy lifting happens here)
        disease, stage, conf = run_prediction(local_path)
        
        # 3. Save to DB
        new_pred = Prediction(
            user_id=current_user.id, plant_id=plant_id, image_filename=permanent_url,
            result=disease, severity=stage, confidence=conf
        )
        db.session.add(new_pred)
        db.session.commit()
        
        # Cleanup local file
        if os.path.exists(local_path): os.remove(local_path)
            
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# --- 5. INITIALIZATION ---
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Using threaded=False for stability on small instances
    app.run(host='0.0.0.0', port=port, threaded=False)