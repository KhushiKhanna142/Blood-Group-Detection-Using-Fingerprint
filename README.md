# Blood-Group-Detection-Using-Fingerprint
A deep learning project that uses a Convolutional Neural Network (CNN) to predict a person’s blood group from fingerprint images. Built with PyTorch and Flask, this web app allows users to upload fingerprint images and get instant blood group predictions.

# Features
- Uses CNN to analyze fingerprint images.
- Supports 8 blood group classes.
- Includes training, validation, and testing pipeline.
- Web interface for uploading fingerprint images and getting predictions.
- Confusion matrix and performance metrics visualization.

## Dataset
The dataset used for training was obtained from Kaggle:
🔗Fingerprint-Based Blood Group Dataset: https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset
Please download it separately and place it in the appropriate folder before training.

## How to Run the Project
- Clone the Repository
git clone https://github.com/KhushiKhanna142/Blood-Group-Detection-Using-Fingerprint.git
cd Blood-Group-Detection-Using-Fingerprint

- Install Dependencies
pip install -r requirements.txt

- Start the Flask App
python run.py

- Visit in Browser
http://localhost:5000

## Model Training
To train the model run:
python Scripts/train_model.py
Ensure the dataset is placed correctly before training.

## 🔐 Authentication
Simple login and registration system using SQLAlchemy and PostgreSQL.
Update the SQLALCHEMY_DATABASE_URI in app/__init__.py with your own credentials.

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

