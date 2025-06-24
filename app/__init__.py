from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)
app.secret_key = 'sanchita'

# DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:a@localhost/fingerprintdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ✅ Import AFTER defining db, login_manager
from app.models import Users

# ✅ Register user_loader
@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# ✅ Import routes LAST (after everything is initialized)
from app import routes
