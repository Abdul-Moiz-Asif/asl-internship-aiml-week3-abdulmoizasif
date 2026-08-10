# AI/ML Internship Week 3: Baseline Supervised Learning

## Project Overview
This project establishes a rigorous, production-ready supervised machine learning workflow using Scikit-Learn. It demonstrates how to properly load data, execute a train/test split to prevent data leakage, encapsulate preprocessing (StandardScaler) and modeling (Logistic Regression) inside a `Pipeline`, and evaluate performance using professional classification metrics.

## Dataset
* **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset (Built-in via `sklearn.datasets`)
* **Details:** 569 instances, 30 numeric predictive features, and a binary target class (Malignant/Benign).

## Setup Instructions
To run this project locally:
1. Clone the repository: `git clone https://github.com/Abdul-Moiz-Asif/asl-internship-aiml-week3-abdulmoizasif.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment (Windows): `.\.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Open `notebook.ipynb` and run all cells.

## Core Results
* **Accuracy:** 97.37% on the isolated test set.
* **Model Stability:** By utilizing a Scikit-Learn `Pipeline`, the `StandardScaler` was strictly fit on the training data, ensuring zero mathematical leakage into the test vault.
* **Performance:** The model produced only 1 False Negative and 2 False Positives, yielding exceptional F1-Scores (0.96 for Malignant, 0.98 for Benign).