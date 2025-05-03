import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data (you may need to run this once)
nltk.download('punkt')
nltk.download('stopwords')

# Load trained models and vectorizer
with open('logistic_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)
with open('tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open('forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', str(text))
    # Convert to lowercase and remove stopwords
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def get_prediction(model, vector):
    prediction = model.predict(vector)[0]
    return "Real" if prediction == 1 else "Fake"

def predict_fake_news(news_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(news_text)
    
    # Vectorize the preprocessed text
    vector = vectorizer.transform([preprocessed_text])
    
    # Make predictions using all models
    predictions = {
        "Logistic Regression": get_prediction(lr_model, vector),
        "Decision Tree": get_prediction(dt_model, vector),
        "Random Forest": get_prediction(rf_model, vector),
        "Naive Bayes": get_prediction(nb_model, vector),
        "SVM": get_prediction(svm_model, vector)
    }
    
    return predictions

# Example usage
if __name__ == "__main__":
    sample_news = """
    Breaking: Scientists have discovered a new planet in our solar system. 
    The planet, temporarily named 'Planet X', is believed to be larger than Jupiter 
    and located beyond Neptune's orbit. More details to follow as the story develops.
    """
    
    results = predict_fake_news(sample_news)
    
    print("Input News:")
    print(sample_news)
    print("\nPredictions:")
    for model, prediction in results.items():
        print(f"{model}: {prediction}")