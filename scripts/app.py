from flask import Flask, request, jsonify
import pickle
import numpy as np 

with open(r"C:\Users\Rishi\Desktop\Data-sci\sentiment analysis pipeline\data\sentiment_model.pkl", "rb") as model_file:
    vectorizer, model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def welcome():
    return """
    <h1>Welcome to Movie Review Sentiment Analysis API</h1>
    <p>This API analyzes the sentiment of movie reviews.</p>
    <p>Use the endpoint <code>/predict</code> to get sentiment predictions for movie reviews.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or "review" not in data:
            return jsonify({"error": "Please provide a review in the request body"}), 400

        review_text = data["review"]
        if not review_text.strip():
            return jsonify({"error": "Review text cannot be empty"}), 400

        # Preprocess text
        review_tfidf = vectorizer.transform([review_text])
        
        # Predict sentiment and convert to standard Python type
        prediction = model.predict(review_tfidf)[0]
        if isinstance(prediction, np.int64):
            prediction = int(prediction) 
        
        return jsonify({
            "sentiment_prediction": prediction,
            "review": review_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)