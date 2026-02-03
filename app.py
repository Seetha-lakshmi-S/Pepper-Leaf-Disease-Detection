from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import pymysql
import cloudinary
import cloudinary.uploader
from markupsafe import Markup
from predict_utils import run_prediction 
from datetime import datetime
from zoneinfo import ZoneInfo

# Allow pymysql to act as MySQLdb for local dev
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# DATABASE SETUP: Detects Render's DATABASE_URL or defaults to local MySQL
db_url = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:Seetha%40123@localhost/pepguard_db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# UPLOAD SETUP (Local temp storage before cloud upload)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- CLOUDINARY CONFIGURATION ---
# Using the credentials you provided
cloudinary.config( 
  cloud_name = "do7xycqmw", 
  api_key = "575413169736574", 
  api_secret = "eCJJemDOYQ17J8jLVcKqS8WTVLo",
  secure = True
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- IST TIMEZONE HELPER ---
def get_ist_time(dt=None):
    if dt is None:
        dt = datetime.now()
    ist = ZoneInfo("Asia/Kolkata")
    return dt.astimezone(ist)

# --- TEMPLATE FILTERS ---
@app.template_filter('ist')
def ist_time_filter(dt):
    if dt is None: return ""
    return get_ist_time(dt).strftime('%d %b %Y, %I:%M %p')

@app.template_filter('ist_date')
def ist_date_filter(dt):
    if dt is None: return ""
    return get_ist_time(dt).strftime('%d %b %Y')

@app.template_filter('nl2br')
def nl2br(value):
    if not value: return ""
    return Markup("<br>".join(str(value).splitlines()))

# --- MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plant_id = db.Column(db.String(150), nullable=False)
    image_filename = db.Column(db.String(500), nullable=False) # Increased length for Cloudinary URLs
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
def load_user(user_id):
    return User.query.get(int(user_id))

# --- DATABASE SEEDER ---
def seed_database_from_json():
    try:
        json_path = os.path.join(app.root_path, 'disease_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                existing = DiseaseInfo.query.filter_by(name=item['name']).first()
                clean = lambda v: "\n".join(v) if isinstance(v, list) else v
                if not existing:
                    db.session.add(DiseaseInfo(
                        name=item['name'],
                        description=clean(item.get('description')),
                        organic_treatment=clean(item.get('organic_treatment')),
                        chemical_treatment=clean(item.get('chemical_treatment')),
                        yield_advice=clean(item.get('yield_advice')),
                        follow_up=clean(item.get('follow_up', 'Monitor regularly.'))
                    ))
            db.session.commit()
    except Exception as e:
        print(f"Seed Error: {e}")

# --- ROUTES ---
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

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('User exists', 'error')
            return redirect(url_for('login'))
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('layout.html')

@app.route('/get_panel/<panel_name>')
@login_required
def get_panel(panel_name):
    if panel_name == 'welcome':
        return render_template('panels/welcome.html')
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
                else: summary['mid'] += 1
        return render_template('panels/summary.html', summary=summary)
    elif panel_name == 'new_diagnosis':
        return render_template('panels/new_diagnosis.html')
    elif panel_name == 'history':
        base = Prediction.query.filter_by(user_id=current_user.id).all()
        history_data = {}
        for p in sorted(base, key=lambda x: x.timestamp, reverse=True):
            if p.plant_id not in history_data: history_data[p.plant_id] = {'entries': []}
            history_data[p.plant_id]['entries'].append(p)
        for plant_id, data in history_data.items():
            data['entries'].sort(key=lambda x: x.timestamp)
            for idx, entry in enumerate(data['entries']): entry.is_followup = (idx != 0)
        return render_template('panels/history.html', history_data=history_data)
    if panel_name.startswith('result/'):
        try:
            pred_id = int(panel_name.split('/')[1])
            return get_result_panel(pred_id)
        except: pass
    return "Panel not found", 404

@app.route('/get_panel/result/<int:prediction_id>')
@login_required
def get_result_panel(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    key = pred.severity if (pred.severity and pred.severity != "Healthy") else pred.result
    info = DiseaseInfo.query.filter_by(name=key).first() or DiseaseInfo.query.first()
    return render_template('panels/result.html', prediction=pred, info=info)

@app.route('/result/<int:prediction_id>')
@login_required
def show_result(prediction_id):
    return redirect(url_for('dashboard') + f'#result-{prediction_id}')

@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    try:
        plant_id = request.form.get('plant_id')
        f = request.files.get('leaf_image')
        if not f: return jsonify({'success': False, 'message': 'No file'})
        
        # 1. Local temp save
        filename = secure_filename(f"{datetime.now().timestamp()}_{f.filename}")
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(local_path)
        
        # 2. Cloudinary Permanent Upload
        upload_result = cloudinary.uploader.upload(local_path, folder="pepguard_uploads")
        permanent_url = upload_result['secure_url']
        
        # 3. AI Prediction (using local file)
        disease, stage, conf = run_prediction(local_path)
        
        # 4. Save to DB with Cloudinary Link
        new_pred = Prediction(
            user_id=current_user.id, plant_id=plant_id, image_filename=permanent_url,
            result=disease, severity=stage, confidence=conf, timestamp=get_ist_time()
        )
        db.session.add(new_pred)
        db.session.commit()
        
        # 5. Clean up local temp file
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return jsonify({'success': True, 'redirect': url_for('show_result', prediction_id=new_pred.id)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# --- STARTUP LOGIC ---
with app.app_context():
    db.create_all()
    seed_database_from_json()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)