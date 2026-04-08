Here is the updated **README.md** file, including the project's folder directory structure based on the provided files.

---

# EduQuery Explorer

**EduQuery Explorer** is a professional-grade student analytics and predictive intelligence platform. Designed for educational administrators and researchers, it transforms complex academic data from 2,000 students across 15 departments into actionable insights through an interactive Streamlit-based dashboard.

## 📂 Project Structure

```text
prompt/
├── .idea/                          # PyCharm project configuration files
│   ├── inspectionProfiles/         # Code inspection settings
│   ├── .gitignore                  # Git ignore rules for IDE files
│   ├── misc.xml                    # Python SDK and project settings
│   ├── modules.xml                 # Project module mapping
│   ├── prompt.iml                  # Project module configuration
│   └── workspace.xml               # IDE workspace state
├── app.py                          # Main Streamlit application logic
├── requirements.txt                # Python dependency list
├── Combined_Education_Dataset.csv  # Raw student dataset (2,000 records)
└── Combined_Education_Summary.csv  # Statistical summary of the dataset
```

## 🚀 Key Features

* **Advanced Data Preprocessing**: Implements a robust automated pipeline including **KNN Imputation** for missing values and **IQR-based Outlier Treatment** to ensure data integrity.
* **Predictive Risk Modeling**: Leverages a **Random Forest Classifier** to identify "At-Risk" students by calculating risk probabilities based on engineered features like engagement scores and attendance patterns.
* **Intelligent Query Engine**: Features a **Natural Language Query (NLQ)** interface that allows users to extract specific datasets and summaries using simple English prompts.
* **Interactive Visualizations**: Includes a dynamic studio for generating heatmaps, distribution plots, and departmental rankings to uncover hidden academic trends.
* **Comprehensive Reporting**: Supports one-click exports of the full dataset to CSV and the generation of professionally formatted **PDF reports** containing automated insights and charts.

## 🛠️ Technical Stack

* **Frontend**: Streamlit
* **Data Processing**: Pandas, NumPy, SciPy
* **Machine Learning**: Scikit-Learn (Random Forest, Gradient Boosting)
* **Visualization**: Matplotlib, Seaborn
* **Reporting**: ReportLab (PDF), OpenPyXL (Excel/CSV)

## 📋 Requirements

The project requires **Python 3.9+** and the following core dependencies:
* `streamlit >= 1.28.0`
* `pandas >= 1.5.0`
* `scikit-learn >= 1.1.0`
* `reportlab >= 3.6.0`

## 📂 Dataset Overview

The system is optimized for the **Combined Educational Intelligence Dataset**, which tracks 17 core columns including:
* **Demographics**: Gender, DOB, State, School Level.
* **Academic Metrics**: GPA, Attendance %, Credits Earned, Program, Department.
* **Behavioral Factors**: Parent Survey results, School Satisfaction, and Absent Days.

---

### 💻 How to Run
1. Navigate to the `prompt/` directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
