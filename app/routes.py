from flask import render_template, redirect, url_for, request, flash, session
from app import app, db
from app.models import Users
from flask_login import login_user, logout_user, current_user, login_required
from app.model_loader import load_model, predict_image
from werkzeug.utils import secure_filename
import os

model = load_model(os.path.join("app", "model", "blood_group_model_best.pth"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']  # Store plain text password
        if Users.query.filter((Users.username==username)|(Users.email==email)).first():
            flash('Username or Email already exists.', 'error')
        else:
            user = Users(username=username, email=email, password=password)  # Store plain text password
            db.session.add(user)
            db.session.commit()
            flash('Registered successfully!', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Fetch the user from the database
        user = Users.query.filter_by(email=email).first()

        if user:
            # Directly compare entered password with stored plain text password
            if user.password == password:
                login_user(user)
                return redirect(url_for('upload'))  # Redirect to the upload page
            else:
                flash("Invalid credentials", "danger")
        else:
            flash("Invalid credentials", "danger")

        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('upload'))

    file = request.files['image']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('upload'))

    filename = secure_filename(file.filename)
    filepath = os.path.join('app/static/uploads', filename)
    file.save(filepath)

    blood_group = predict_image(model, filepath)
    session['prediction'] = blood_group
    session['image_path'] = filename
    return redirect(url_for('result'))

@app.route('/result')
@login_required
def result():
    return render_template('result.html', prediction=session.get('prediction'), image_path=session.get('image_path'))

