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

├── app/ # Flask application code (routes, models, templates, static files)
  │ ├── init.py
  │ ├── routes.py
  │ ├── model_loader.py
  │ ├── models.py
  │ ├── static/
  │ └── templates/

├── scripts/ # Training and utility scripts
  │ └── train_model.py

├── models/ # Saved trained models (e.g. blood_group_model_best.pth)
├── cnn_model.py # CNN model architecture definition
├── run.py # Flask app entry point to start the server
├── LICENSE
├── README.md
└── pycache/ # Auto-generated Python cache files (can be ignored)





