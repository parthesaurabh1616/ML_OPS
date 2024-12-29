from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Define a data model for incoming data
class Tweet(BaseModel):
    text: str

# Initialize the FastAPI application
app = FastAPI()

# Load the trained model and vectorizer
classifier = joblib.load("sentiment_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    # Preprocess and vectorize the text
    text_features = vectorizer.transform([tweet.text])
    # Predict the sentiment
    prediction = classifier.predict(text_features)
    # Return the prediction as a JSON response
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return {"text": tweet.text, "sentiment": sentiment}
