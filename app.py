from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import numpy as np
import pandas as pd
import warnings
import sklearn
import mysql.connector
from mysql.connector import Error
import re
from functools import wraps
import datetime
import os

warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# MySQL Configuration - UPDATE THESE FOR PYTHONANYWHERE!
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'root'),
    'database': os.environ.get('DB_NAME', 'heart'),
    'auth_plugin': 'mysql_native_password'
}

def get_db_connection():
    """Create and return a MySQL connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def load_model_safely():
    """Load model with version handling"""
    try:
        import joblib
        try:
            # Use absolute paths for deployment
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'model', 'random_forest_model.joblib')
            scaler_path = os.path.join(base_dir, 'model', 'scaler.joblib')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Loaded with joblib successfully")
            return model, scaler
        except Exception as e:
            print(f"Joblib failed: {e}, trying pickle...")
            import pickle
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            print("Loaded with pickle successfully")
            return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, scaler = load_model_safely()

if model is None or scaler is None:
    print("WARNING: Could not load model. Please check model files.")

if hasattr(model, 'feature_names_in_'):
    EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    EXPECTED_COLUMNS = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        'age_chol_interaction', 'age_group_encoded'
    ]

def create_engineered_features(age, chol):
    """Recreate the engineered features"""
    age_chol_interaction = age * chol
    
    if age < 40:
        age_group_encoded = 0  # Young
    elif age < 60:
        age_group_encoded = 1  # Middle-Aged
    else:
        age_group_encoded = 2  # Senior
    
    return age_chol_interaction, age_group_encoded

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loggedin' not in session:
            flash('Please login to access this page!', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication Routes
@app.route('/')
def index():
    """Redirect to signup page as the first page"""
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return redirect(url_for('signup'))
        
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!', 'error')
            return redirect(url_for('signup'))
        
        try:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor()
                # Check if user already exists
                cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
                if cursor.fetchone():
                    flash('Username or email already exists!', 'error')
                    return redirect(url_for('signup'))
                
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, password)
                )
                connection.commit()
                cursor.close()
                connection.close()
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
        except Error as e:
            print(f"Database error: {e}")
            flash('An error occurred during registration!', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(
                    "SELECT * FROM users WHERE username = %s AND password = %s",
                    (username, password)
                )
                user = cursor.fetchone()
                cursor.close()
                connection.close()
                
                if user:
                    session['loggedin'] = True
                    session['id'] = user['id']
                    session['username'] = user['username']
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Incorrect username or password!', 'error')
                    return redirect(url_for('login'))
        except Error as e:
            flash('An error occurred during login!', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

# Dashboard and Application Routes
@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page with statistics"""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            
            # Get total predictions
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = %s", (session['id'],))
            total_predictions = cursor.fetchone()['count']
            
            # Get positive cases
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = %s AND prediction_result = 'Heart Disease Detected'", (session['id'],))
            positive_cases = cursor.fetchone()['count']
            
            # Get recent activity (today's predictions)
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = %s AND DATE(created_at) = CURDATE()", (session['id'],))
            recent_activity = cursor.fetchone()['count']
            
            cursor.close()
            connection.close()
            
            # Calculate accuracy rate (using your model's accuracy)
            accuracy_rate = 88.52  # Your model's accuracy
            
            return render_template('dashboard.html',
                                 username=session['username'],
                                 total_predictions=total_predictions,
                                 positive_cases=positive_cases,
                                 accuracy_rate=accuracy_rate,
                                 recent_activity=recent_activity)
    except Error as e:
        print(f"Database error: {e}")
        # Return default values if there's an error
        return render_template('dashboard.html',
                             username=session['username'],
                             total_predictions=0,
                             positive_cases=0,
                             accuracy_rate=88.52,
                             recent_activity=0)

@app.route('/new_prediction')
@login_required
def new_prediction():
    """New prediction page"""
    return render_template('index.html', username=session['username'])

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Heart disease prediction"""
    if model is None or scaler is None:
        return render_template('result.html', 
                             result="Error: Model not loaded properly", 
                             confidence=0,
                             features={},
                             username=session['username'])
    
    try:
        age = float(request.form['age'])
        chol = float(request.form['chol'])
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        
        age_chol_interaction, age_group_encoded = create_engineered_features(age, chol)
        
        data = {
            'age': age,
            'sex': float(request.form['sex']),
            'cp': float(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': chol,
            'fbs': float(request.form['fbs']),
            'restecg': float(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': float(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': float(request.form['slope']),
            'ca': float(request.form['ca']),
            'thal': float(request.form['thal']),
            'age_chol_interaction': age_chol_interaction,
            'age_group_encoded': age_group_encoded
        }
        
            # Create features in the EXACT same order as training
        features_list = [data[col] for col in EXPECTED_COLUMNS]
        features_array = np.array(features_list).reshape(1, -1)
        
        # Scale the features - this will work without feature names warning
        features_scaled = scaler.transform(features_array)
        
        # Create DataFrame with correct feature names and order for prediction
        features_scaled_df = pd.DataFrame(features_scaled, columns=EXPECTED_COLUMNS)
        
        # Make prediction
        prediction = model.predict(features_scaled_df)
        probability = model.predict_proba(features_scaled_df)
        
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]
        confidence_percent = round(confidence * 100, 2)
        
        display_features = {k: v for k, v in data.items() if k not in ['age_chol_interaction', 'age_group_encoded']}
        
        # Save prediction to database - Convert numpy types to Python native types
        try:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor()
                cursor.execute('''
                    INSERT INTO predictions 
                    (user_id, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    session['id'], 
                    patient_name, 
                    int(data['age']),          # Convert to Python int
                    int(data['sex']),          # Convert to Python int
                    int(data['cp']),           # Convert to Python int
                    int(data['trestbps']),     # Convert to Python int
                    int(data['chol']),         # Convert to Python int
                    int(data['fbs']),          # Convert to Python int
                    int(data['restecg']),      # Convert to Python int
                    int(data['thalach']),      # Convert to Python int
                    int(data['exang']),        # Convert to Python int
                    float(data['oldpeak']),    # Convert to Python float
                    int(data['slope']),        # Convert to Python int
                    int(data['ca']),           # Convert to Python int
                    int(data['thal']),         # Convert to Python int
                    result, 
                    float(confidence_percent)  # Convert to Python float
                ))
                connection.commit()
                cursor.close()
                connection.close()
                print("Prediction saved to database successfully")
        except Error as e:
            print(f"Error saving prediction to database: {e}")
        
        return render_template('result.html', 
                             result=result, 
                             confidence=confidence_percent,
                             features=display_features,
                             username=session['username'])
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('result.html', 
                             result=f"Error: {str(e)}", 
                             confidence=0,
                             features={},
                             username=session['username'])

if __name__ == '__main__':
    print(f"scikit-learn version: {sklearn.__version__}")
    app.run(debug=False)  # Set debug=False for production