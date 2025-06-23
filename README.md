# 🧠 Neural Network Image Recognition (CIFAR-10)

This project is a simple Convolutional Neural Network (CNN) implemented in Python using TensorFlow. It performs image classification on the CIFAR-10 dataset, which contains 60,000 color images in 10 different classes.

## 🚀 Features

- Image classification using CNN
- Trained on CIFAR-10 dataset
- Visualizes predictions using Matplotlib
- Achieves around 65–70% accuracy on test set

## 🧰 Tech Stack

- Python 3.10
- TensorFlow 2.x
- Matplotlib
- NumPy

## 📦 Dataset

[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## 📁 Project Structure

NN.py # Main Python script
README.md # Project description


## ▶️ How to Run

### 1. Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create virtual environment (optional):
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install tensorflow matplotlib

5. Run the script:
python NN.py

📷 Output
The model trains for 10 epochs
Displays 5 random images from the test set with predicted labels
