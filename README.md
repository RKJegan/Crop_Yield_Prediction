Tamil Nadu Crop Production Level Prediction

A Machine Learning project that predicts crop production level (Low / Medium / High) in Tamil Nadu using a Decision Tree Classifier.

📌 Project Overview

This project uses historical crop yield data to classify production into three levels:

Low

Medium

High

The classification is based on features like:

State Name

District Name

Crop Year

Season

Crop

Area

The model is trained using a Decision Tree algorithm from scikit-learn.

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

scikit-learn

📂 Dataset

File: TN yield data.csv

The dataset contains crop production details for Tamil Nadu.

Missing values in the Production column are removed before training.

⚙️ Project Workflow
1️⃣ Data Preprocessing

Load dataset using Pandas

Remove missing production values

Convert Production into 3 classes using quantile-based binning:

Low

Medium

High

Encode categorical variables using LabelEncoder

2️⃣ Feature Selection

Input Features (X):

State_Name
District_Name
Crop_Year
Season
Crop
Area


Target (Y):

Production_level

3️⃣ Train-Test Split

80% Training Data

20% Testing Data

Random State = 42

4️⃣ Model Building

Algorithm Used:

DecisionTreeClassifier

Max Depth = 10

5️⃣ Model Evaluation

Accuracy Score

Confusion Matrix

Classification Report

6️⃣ Decision Tree Visualization

The trained tree is visualized using plot_tree() from scikit-learn.

📊 Model Performance Metrics

The model prints:

Accuracy Score

Confusion Matrix

Precision

Recall

F1-Score

🧪 User Input Prediction

After training, the program allows user input:

Enter District
Enter Crop Year
Enter Season
Enter Crop Name
Enter Area


⚠️ Inputs are case sensitive

The model encodes the inputs and predicts:

Predicted Production Level: Low / Medium / High

▶️ How to Run the Project

Install required libraries:

pip install pandas numpy matplotlib scikit-learn


Place TN yield data.csv in the same directory as the script.

Run the Python file:

python DSimp.py


Provide inputs when prompted.

📁 Project Structure
├── DSimp.py
├── TN yield data.csv
└── README.md

📈 Future Improvements

Use Random Forest or XGBoost for better accuracy

Deploy using Flask or Streamlit

Add feature importance analysis

Improve handling of unseen categorical inputs

👤 Author

Machine Learning project for crop production classification using Decision Tree.
