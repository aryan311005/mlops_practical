from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open("model_v1.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "MLOps CI/CD Working!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
