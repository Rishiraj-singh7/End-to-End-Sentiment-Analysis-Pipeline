## Project Setup
**Install dependencies:</u>**
</u>pip install -r requirements.txt</u>

Data Acquisition
**Dataset**: Downloaded from Kaggle (IMDB Dataset).
Run Instructions

**Train the Model:**
<u>python scripts/train_model.py</u>

**Start Flask Server:**
<u>python scripts/app.py</u>  

**Test Endpoints in Postman or Browser:**
**Welcome Page:** http://127.0.0.1:5000/
**Prediction Endpoint:** http://127.0.0.1:5000/predict
Example Body:

{"text": "This movie is fantastic!"}  

**Model Info**
**Model:** Logistic Regression
**Accuracy:** 98% on the test set.
