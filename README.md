# FlipkartSentimentAnalysis
# 🛒 Flipkart Product Review Sentiment Analysis

> **Unlocking customer insights using Natural Language Processing (NLP) and Machine Learning.**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge)

## 🌟 Executive Summary

In the e-commerce world, customer feedback is gold. This project is a comprehensive **Sentiment Analysis pipeline** designed to classify Flipkart product reviews into **Positive**, **Negative**, or **Neutral** sentiments.

By leveraging **VADER (Valence Aware Dictionary and sEntiment Reasoner)** for automatic labeling and **Decision Tree Classifiers** for prediction, this model achieves an outstanding accuracy rate, helping businesses understand user satisfaction instantly.

## 📊 Project Workflow

The notebook follows a rigorous data science lifecycle:

1.  **Data Ingestion:** Loading large-scale review datasets (`Dataset-SA.csv`).
2.  **Text Preprocessing:** Cleaning raw text by removing punctuation, special characters, and stopwords to reduce noise.
3.  **Sentiment Labeling (VADER):** * Utilizes the `SentimentIntensityAnalyzer` to assign polarity scores (Positive/Negative/Neutral) to every review summary.
    * *Logic:* If Positive > Negative → Label 1; if Negative > Positive → Label -1; else Label 0.
4.  **Feature Extraction:** Converts textual data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
5.  **Model Training:** Trains a **Decision Tree Classifier** on the processed data.

## 🚀 Key Results & Performance

The model demonstrates exceptional performance on the testing dataset:

* **🏆 Overall Accuracy:** **~99.45%**
* **Precision & Recall:** Near perfect scores across all sentiment classes.
* **Confusion Matrix:** Shows minimal misclassification between positive and negative reviews.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.45% |
| **F1-Score (Positive)** | 0.99 |
| **F1-Score (Negative)** | 0.96 |

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **NLP & ML:** NLTK, Scikit-Learn (DecisionTree, TfidfVectorizer)

## 📈 Visualizations

The project includes insightful visualizations to understand data distribution:
* **Sentiment Distribution Pie Chart:** A visual breakdown of 5-star vs 1-star ratings.
* **Confusion Matrix Heatmap:** A graphical representation of the model's true positives vs false positives.

## ⚙️ How to Run

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/flipkart-sentiment-analysis.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn plotly wordcloud
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook Flipkartsentiment.ipynb
    ```
4.  **Execute Cells:** Run the cells sequentially to preprocess data, train the model, and view the results.

## 🔮 Future Enhancements

* [ ] Implement Deep Learning models (LSTM/BERT) for comparison.
* [ ] Create a Flask/Streamlit web app for real-time user input.
* [ ] Expand the dataset to include other e-commerce platforms like Amazon.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1.  Fork the Project
2.  Create your Feature Branch
3.  Commit your Changes
4.  Open a Pull Request

---

Thank You For Visiting!!

Analyzed with 💻 by Shankari N
