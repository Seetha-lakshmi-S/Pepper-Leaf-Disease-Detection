import os
import json
import gc
import re
import pymysql
import cloudinary
import cloudinary.uploader
import base64
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from markupsafe import Markup
from datetime import datetime
from zoneinfo import ZoneInfo

from predict_utils import run_prediction 

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# --- 1. CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')
db_url = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:Seetha%40123@localhost/pepguard_db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
    "pool_size": 5,
    "max_overflow": 2
}

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

cloudinary.config( 
    cloud_name = "do7xycqmw", 
    api_key = "575413169736574", 
    api_secret = "eCJJemDOYQ17J8jLVcKqS8WTVLo",
    secure = True
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- 2. HELPERS ---
def strip_citations(text):
    if not text: return ""
    
    text = re.sub(r'\[cite_start\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[cite_end\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:cite|source):\s*\d+(?:\s*,\s*\d+)*\]', '', text, flags=re.IGNORECASE)
    
    breaks = [
        'Today:', 'Weekly:', 'Daily:', 'Monthly:', 'Always:', 'Immediately:', 
        'Evening or early morning:', 'As soon as possible:', 'Flowering stage:',
        'Day 1:', 'Day 3–4:', 'Day 5:', 'Day 10:', 'After 5 days:', 'Repeat interval:',
        'Every 4–5 days:', 'Next season:', 'Stop when:',
        'Frequency:', 'What to monitor:', 'Actions:', 'Notes:', 
        'Actions if stable or improving:', 'Actions if improving:', 
        'Actions if worsening:', 'Actions if not improving:', 
        'Actions if not improving or spreading fast:',
        'If sucking pests are present:', 'If soft rot or bacterial ooze is present:',
        'If soft rot is present:', '•'
    ]
    
    breaks.sort(key=len, reverse=True)
    
    for item in breaks:
        pattern = f"(?!^){re.escape(item)}"
        text = re.sub(pattern, f"\n{item}", text)
    
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def get_ist_time(dt=None):
    if dt is None: dt = datetime.now()
    ist = ZoneInfo("Asia/Kolkata")
    return dt.astimezone(ist)

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
    timestamp = db.Column(db.DateTime, default=lambda: get_ist_time())

class DiseaseInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    organic_treatment = db.Column(db.Text)
    chemical_treatment = db.Column(db.Text)
    yield_advice = db.Column(db.Text)
    follow_up = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id): return db.session.get(User, int(user_id))

# --- 4. CLOUD INITIALIZATION ---
def seed_database_from_json():
    try:
        json_path = os.path.join(app.root_path, 'disease_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                existing_disease = DiseaseInfo.query.filter_by(name=item['name']).first()
                
                if existing_disease:
                    existing_disease.description = item.get('description')
                    existing_disease.organic_treatment = item.get('organic_treatment')
                    existing_disease.chemical_treatment = item.get('chemical_treatment')
                    existing_disease.yield_advice = item.get('yield_advice')
                    existing_disease.follow_up = item.get('follow_up')
                else:
                    db.session.add(DiseaseInfo(
                        name=item['name'],
                        description=item.get('description'),
                        organic_treatment=item.get('organic_treatment'),
                        chemical_treatment=item.get('chemical_treatment'),
                        yield_advice=item.get('yield_advice'),
                        follow_up=item.get('follow_up')
                    ))
            
            db.session.commit()
            print("Database successfully synchronized with disease_data.json")
    except Exception as e: 
        print(f"Seed Error: {e}")

@app.before_request
def create_tables():
    app.before_request_funcs[None].remove(create_tables)
    db.create_all()
    seed_database_from_json()

# --- 5. ROUTES ---
@app.route('/get_panel/<path:panel_name>')
@login_required
def get_panel(panel_name):
    if panel_name == 'welcome': return render_template('panels/welcome.html')
    elif panel_name == 'summary':
        all_preds = Prediction.query.filter_by(user_id=current_user.id).all()
        latest = {p.plant_id: p for p in all_preds}
        summary = {"total_plants": len(latest), "healthy": 0, "early": 0, "mid": 0, "advanced": 0, "diseased_total": 0}
        for p in latest.values():
            if p.result == 'Healthy': summary['healthy'] += 1
            else:
                summary['diseased_total'] += 1
                sev = (p.severity or "").lower()
                if 'early' in sev: summary['early'] += 1
                elif 'mid' in sev: summary['mid'] += 1
                elif 'advanced' in sev: summary['advanced'] += 1
        return render_template('panels/summary.html', summary=summary)
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
        # Fallback logic for fetching disease info
        info = DiseaseInfo.query.filter_by(name=pred.severity).first() or \
               DiseaseInfo.query.filter_by(name=pred.result).first() or \
               DiseaseInfo.query.filter_by(name="Bacterial Disease").first()
        return render_template('panels/result.html', prediction=pred, info=info)
    return "Panel not found", 404

@app.route('/')
def landing():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('login'))
        hashed_pw = generate_password_hash(request.form.get('password'), method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('layout.html')

@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    local_path = None
    try:
        f = request.files.get('leaf_image')
        if not f: return jsonify({'success': False, 'message': 'No file'})
        filename = secure_filename(f"{datetime.now().timestamp()}_{f.filename}")
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(local_path)
        
        disease, stage, conf = run_prediction(local_path)
        upload_result = cloudinary.uploader.upload(local_path, folder="pepguard_uploads")
        
        new_pred = Prediction(
            user_id=current_user.id, 
            plant_id=request.form.get('plant_id', 'Unknown'), 
            image_filename=upload_result['secure_url'], 
            result=disease, 
            severity=stage, 
            confidence=conf
        )
        db.session.add(new_pred)
        db.session.commit()
        return jsonify({'success': True, 'redirect': url_for('show_result', prediction_id=new_pred.id)})
    finally:
        if local_path and os.path.exists(local_path): os.remove(local_path)
        gc.collect()

@app.route('/result/<int:prediction_id>')
@login_required
def show_result(prediction_id):
    return redirect(url_for('dashboard') + f'#result-{prediction_id}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)