from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
import io
import cv2
import joblib
import math

app = Flask(__name__)


class CustomActivation(Layer):
    def __init__(self, num_features, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.num_features = num_features
        self.w = self.add_weight(
            shape=(num_features,), initializer="ones", trainable=True
        )

    def call(self, inputs):
        weighted_sum = K.sum(inputs * self.w, axis=-1)
        divided_sum = weighted_sum / 8.0
        return K.sigmoid(divided_sum)

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features})
        return config


abnormality_model = joblib.load("Models/decision_tree_anomaly_detection.pkl")

self_awareness_model = joblib.load("Models/self_awareness_model.pkl")

risk_assessment_model = load_model(
    "Models/risk_model.h5",
    custom_objects={"CustomActivation": CustomActivation},
)

ecg_model = load_model("Models/Resnet.h5")


abnormality_class_mapping = {0: "Normal Vitals", 1: "Abnormal Vitals"}


self_awareness_class_mapping = {
    0: "Sitting",
    1: "Walking",
    2: "Running",
    3: "Sleeping",
    4: "No Activity",
}


ecg_class_mapping = {
    0: "Supraventricular Premature Beat",
    1: "Myocardial Infarction",
    2: "Unclassifiable Beat",
    3: "Normal Beat",
    4: "Fusion of Ventricular and Normal Beat",
    5: "Premature Ventricular Contraction",
}

ecg_severity_mapping = {
    "Supraventricular Premature Beat": 70,
    "Myocardial Infarction": 95,
    "Unclassifiable Beat": 35,
    "Normal Beat": 0,
    "Fusion of Ventricular and Normal Beat": 60,
    "Premature Ventricular Contraction": 80,
}


def preprocess_image(img_stream):
    # Load the image with target size
    img = image.load_img(img_stream, target_size=(224, 224), color_mode="grayscale")
    img = image.img_to_array(img)

    # Convert grayscale image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Rescale the image
    img = img / 255.0

    # Expand dimensions to match the model's input shape
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def generate_lis():
    try:
        # Handle file upload
        ecg_image = request.files["ecg_file"]
        img_stream = io.BytesIO(ecg_image.read())
        processed_img = preprocess_image(img_stream)

        vital_signs_features = [
            float(request.form.get("respiratory_rate")),
            float(request.form.get("blood_oxygen")),
            float(request.form.get("blood_sugar")),
            float(request.form.get("systolic_bp")),
            float(request.form.get("diastolic_bp")),
            float(request.form.get("heart_rate")),
        ]

        risk_factor_features = np.array(
            [
                int(request.form.get("cholesterol")),
                int(request.form.get("diabetes")),
                int(request.form.get("family_history")),
                int(request.form.get("hypertension")),
                int(request.form.get("physical_inactivity")),
            ]
        ).reshape(1, -1)

        # Example of handling predictions (adjust based on actual model functions)
        vital_prediction = abnormality_model.predict([vital_signs_features])
        vital_prediction = abnormality_class_mapping[int(vital_prediction[0])]

        self_awareness_prediction = self_awareness_model.predict([vital_signs_features])
        self_awareness_prediction = self_awareness_class_mapping[
            int(self_awareness_prediction[0])
        ]

        risk_prediction = risk_assessment_model.predict([risk_factor_features])
        risk_prediction = risk_prediction[0]

        ecg_prediction = ecg_model.predict(processed_img)
        ecg_prediction = np.argmax(ecg_prediction, axis=1)
        ecg_prediction = ecg_class_mapping[ecg_prediction[0]]

        lis_score = (
            (
                (risk_prediction * 100) * 0.4
                + (ecg_severity_mapping[ecg_prediction]) * 0.6
            )
            * 12
        ) / 100

        return jsonify(
            {
                "abnormality_prediction": str(vital_prediction),
                "self_awareness_prediction": str(self_awareness_prediction),
                "risk_prediction": str(round(risk_prediction, 3)),
                "ecg_prediction": str(ecg_prediction),
                "lis_score": str(math.ceil(lis_score)),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
