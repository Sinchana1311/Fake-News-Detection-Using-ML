This is a web-based fake news detection system built using machine learning algorithms. The project processes news text data to classify articles as real or fake, with a complete workflow from data preprocessing to model deployment using Flask and frontend technologies.

🔧 Software Requirements
Programming Language: Python

Web Framework (Backend): Flask

Frontend Technologies: HTML, CSS, JavaScript

Libraries Used:

scikit-learn

numpy

pandas

nltk

matplotlib, seaborn (for data visualization)

📂 Dataset
Source: Kaggle – Fake and Real News Dataset

File Used: train.csv

💡 Machine Learning Algorithms Used
Decision Tree Classifier

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

Naive Bayes Classifier

⚙️ Project Workflow
Data Loading (train.csv)

Data Cleaning and Preprocessing

Text Processing using NLTK

Data Visualization

Feature Extraction (TF-IDF)

Train-Test Splitting

Model Training (All 5 Algorithms)

Model Prediction

Model Evaluation

*Accuracy

*Precision

*Recall

*F1 Score

*Classification Report

*Confusion Matrix

Result Visualization

🔮 Future Enhancements
Integrate deep learning models such as LSTM, GRU, or BERT for improved accuracy.

Implement real-time news scraping from APIs (e.g., newsapi.org) for live predictions.

Add user authentication and save prediction history.

Extend to multi-language support using multilingual NLP models.

Improve UI/UX with modern frontend frameworks like React or Bootstrap.

Deploy the project on cloud platforms like Heroku, AWS, or Render for public access.

🧾 Conclusion
This project successfully demonstrates the use of machine learning and natural language processing to identify and classify fake news articles. It integrates multiple classification algorithms, evaluates their performance, and presents results through a simple web interface. With further enhancements, this system can be transformed into a robust tool for combating misinformation in real time.

