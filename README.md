# Blood-Group-Detection-Using-Fingerprint
A deep learning project that uses a Convolutional Neural Network (CNN) to predict a person’s blood group from fingerprint images. Built with PyTorch and Flask, this web app allows users to upload fingerprint images and get instant blood group predictions.

# Features
- Uses CNN to analyze fingerprint images.
- Supports 8 blood group classes.
- Includes training, validation, and testing pipeline.
- Web interface for uploading fingerprint images and getting predictions.
- Confusion matrix and performance metrics visualization.

## Project Structure
project-root/
├── Scripts/
│   └── train_model.py
├── app/
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── model_loader.py
│   ├── models.py
│   ├── routes.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── images/
│   │   │   └── fingerprint.jpg
│   │   └── uploads/
│   ├── templates/
│   │   ├── about.html
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── result.html
│   │   └── upload.html
│   └── __pycache__/          # Auto-generated Python bytecode files
│       ├── __init__.cpython-313.pyc
│       ├── model_loader.cpython-313.pyc
│       ├── models.cpython-313.pyc
│       └── routes.cpython-313.pyc
├── model/
│   ├── blood_group_model_best.pth
│   └── class_names.txt
├── LICENSE
├── README.md
└── run.py
