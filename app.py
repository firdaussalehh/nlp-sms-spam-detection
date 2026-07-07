import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

model = pickle.load(open("sms_spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

df = pd.read_csv("sms_dataset_cleaned.csv")
metrics_df = pd.read_csv("evaluation_metrics.csv")
cm_df = pd.read_csv("confusion_matrix.csv", index_col=0)

st.set_page_config(
    page_title="SMS Spam Detection Dashboard",
    page_icon="📩",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4e79;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 25px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}
.result-spam {
    padding: 25px;
    border-radius: 18px;
    background-color: #ffecec;
    color: #b00020;
    font-size: 26px;
    font-weight: 700;
}
.result-ham {
    padding: 25px;
    border-radius: 18px;
    background-color: #eafaf1;
    color: #0b7a3b;
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "SMS Spam Detector",
        "Dataset Explorer",
        "Evaluation Metrics",
        "Confusion Matrix",
        "Prediction History",
        "About Project"
    ]
)

if menu == "Home":
    st.markdown('<div class="big-title">📩 SMS Spam Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detect whether an SMS message is Spam or Ham using Natural Language Processing.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "SMS Spam Collection")
    col2.metric("Model", "Logistic Regression")
    col3.metric("Feature Extraction", "TF-IDF")

    st.write("")
    st.subheader("Project Workflow")
    st.markdown("""
    1. User enters an SMS message  
    2. Text is cleaned and preprocessed  
    3. TF-IDF converts text into numerical features  
    4. Logistic Regression classifies the message  
    5. The system displays Spam or Ham prediction  
    """)

elif menu == "SMS Spam Detector":
    st.title("🔍 SMS Spam Detector")
    st.write("Enter an SMS message below and the system will classify it as Spam or Ham.")

    sms_text = st.text_area(
        "Enter SMS message:",
        height=180,
        placeholder="Type SMS message here..."
    )

    if st.button("Detect Message"):
        if sms_text.strip() == "":
            st.warning("Please enter an SMS message.")
        else:
            cleaned = clean_text(sms_text)
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            probability = model.predict_proba(transformed)[0]
            confidence = max(probability) * 100

            result = "Spam" if prediction == 1 else "Ham"

            st.session_state.history.append({
                "SMS Message": sms_text,
                "Prediction": result,
                "Confidence (%)": round(confidence, 2)
            })

            if result == "Spam":
                st.markdown(f'<div class="result-spam">🚨 Prediction: Spam</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-ham">✅ Prediction: Ham</div>', unsafe_allow_html=True)

            st.metric("Confidence Score", f"{confidence:.2f}%")

elif menu == "Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.write("This section shows the SMS Spam Collection Dataset used in this project.")

    col1, col2 = st.columns(2)
    col1.metric("Total Messages", len(df))
    col2.metric("Classes", "Ham / Spam")

    st.subheader("Dataset Preview")
    st.dataframe(df[["label", "message"]].head(100), use_container_width=True)

    st.subheader("Class Distribution")

    label_count = df["label"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(label_count.index, label_count.values)
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Number of Messages")
    st.pyplot(fig)

elif menu == "Evaluation Metrics":
    st.title("📊 Evaluation Metrics")
    st.write("The model performance was evaluated using Accuracy, Precision, Recall, and F1-score.")

    accuracy = metrics_df.loc[metrics_df["Metric"] == "Accuracy", "Score"].values[0]
    precision = metrics_df.loc[metrics_df["Metric"] == "Precision", "Score"].values[0]
    recall = metrics_df.loc[metrics_df["Metric"] == "Recall", "Score"].values[0]
    f1 = metrics_df.loc[metrics_df["Metric"] == "F1 Score", "Score"].values[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")
    col4.metric("F1 Score", f"{f1*100:.2f}%")

    st.subheader("Evaluation Chart")

    fig, ax = plt.subplots()
    ax.bar(metrics_df["Metric"], metrics_df["Score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    st.pyplot(fig)

elif menu == "Confusion Matrix":
    st.title("📈 Confusion Matrix")
    st.write("The confusion matrix shows correct and incorrect predictions.")

    st.dataframe(cm_df, use_container_width=True)

    fig, ax = plt.subplots()
    ax.imshow(cm_df.values)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)

    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            ax.text(j, i, cm_df.values[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif menu == "Prediction History":
    st.title("📋 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        st.subheader("Prediction Summary")

        prediction_count = history_df["Prediction"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(prediction_count, labels=prediction_count.index, autopct="%1.1f%%")
        st.pyplot(fig)

elif menu == "About Project":
    st.title("ℹ️ About Project")

    st.markdown("""
    ## SMS Spam Detection Dashboard Using Natural Language Processing

    ### Problem Statement
    Mobile users often receive unwanted SMS messages such as scams, advertisements, fake prizes, and suspicious links. These spam messages can disturb users and may expose them to fraud or phishing attacks. Therefore, an automated system is needed to classify SMS messages as either Spam or Ham.

    ### Objectives
    - To develop an SMS spam detection system.
    - To apply Natural Language Processing techniques for text classification.
    - To classify SMS messages into Spam or Ham.
    - To evaluate model performance using Accuracy, Precision, Recall, and F1-score.
    - To display prediction results through an interactive dashboard.

    ### Methodology
    - Dataset: SMS Spam Collection Dataset
    - Text preprocessing: lowercase conversion, cleaning, punctuation removal
    - Feature extraction: TF-IDF Vectorizer
    - Model: Logistic Regression
    - Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix

    ### Technologies
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Matplotlib

    ### Future Work
    The system can be improved by adding multilingual SMS detection, larger datasets, and deep learning models such as LSTM or BERT.
    """)
