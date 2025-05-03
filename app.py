from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

app = Flask(__name__)

# Dummy data generation
def generate_dummy_data():
    num_samples = 1000
    num_features = 10
    num_test_samples = 200

    x_train = np.random.rand(num_samples, num_features)
    y_train = np.random.randint(2, size=num_samples)

    x_test = np.random.rand(num_test_samples, num_features)
    y_test = np.random.randint(2, size=num_test_samples)

    return x_train, y_train, x_test, y_test

def label_to_text(label):
    return "Real" if label == 1 else "Fake"

# Initialize models
x_train, y_train, x_test, y_test = generate_dummy_data()
lr_model = LogisticRegression().fit(x_train, y_train)
dt_model = DecisionTreeClassifier().fit(x_train, y_train)
rf_model = RandomForestClassifier().fit(x_train, y_train)
nb_model = MultinomialNB().fit(np.abs(x_train), y_train)
svm_model = SVC(kernel='linear', random_state=42).fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    news_text = ""
    lr_prediction = dt_prediction = rf_prediction = nb_prediction = svm_prediction = None

    if request.method == 'POST':
        news_text = request.form.get('news_text', '')

        if news_text:
            # Here, you should convert the news text into the appropriate feature vector.
            # For simplicity, let's assume you have a function convert_text_to_features for that.
            # feature_vector = convert_text_to_features(news_text)

            # Dummy feature vector for example purposes
            feature_vector = np.random.rand(1, 10)

            lr_prediction = label_to_text(lr_model.predict(feature_vector)[0])
            dt_prediction = label_to_text(dt_model.predict(feature_vector)[0])
            rf_prediction = label_to_text(rf_model.predict(feature_vector)[0])
            nb_prediction = label_to_text(nb_model.predict(np.abs(feature_vector))[0])
            svm_prediction = label_to_text(svm_model.predict(feature_vector)[0])

    return render_template('predict.html', 
                           news_text=news_text, 
                           lr_prediction=lr_prediction, 
                           dt_prediction=dt_prediction,
                           rf_prediction=rf_prediction, 
                           nb_prediction=nb_prediction, 
                           svm_prediction=svm_prediction)

if __name__ == '__main__':
    app.run(debug=True)
