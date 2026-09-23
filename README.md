# defect_triage

Installation
Clone the Repository
 
git clone https://github.com/emb-doreswamy/defect_triage.git
 
cd automotive-defect-triage-ai
Create Virtual Environment (Optional)
 
 
python -m venv venv
 
**Activate Virtual Environment**
venv\Scripts\activate
 
**Linux / macOS**
source venv/bin/activate
 
**Install Required Libraries**
pip install streamlit

pip install pandas

pip install numpy
 
pip install scikit-learn
 
pip install catboost
 
pip install sentence-transformers
 
pip install plotly
 
pip install openpyxl
 
pip install joblib
 
**Or Install Everything at Once**
pip install streamlit pandas numpy scikit-learn catboost sentence-transformers plotly openpyxl joblib
 
**Running the Project**

**Step 1: Train the Model**
python Training.py
 
**This generates:**
catboost_defect_triage_model.pkl 

label_encoder.pkl

feature_importance.xlsx
prediction_comparison.xlsx
vehicle.log
 
**Step 2: Launch Dashboard**
streamlit run Dashboard.py
 
**Step 3: Open Application**
http://localhost:8501
 
**Tool Stack**

**Machine Learning & NLP**
CatBoost Classifier
Sentence Transformers (MiniLM)
Scikit-Learn

**Data Processing**
Pandas
NumPy

**Dashboard & Visualization**
Streamlit
Plotly

**Model Persistence**
Joblib

**Excel Processing**
OpenPyXL

**Project Structure**
 automotive-defect-triage-ai/

├── Dashboard.py

├── Training.py

├── automotive_defect_triage_balanced.xlsx

├── catboost_defect_triage_model.pkl

├── label_encoder.pkl

├── embedding_config.pkl

├── feature_importance.xlsx

├── prediction_comparison.xlsx

├── prediction_history.xlsx

├── vehicle.log

├── Embitel_logo.png

└── README.md
