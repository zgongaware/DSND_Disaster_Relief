import pandas as pd
import pickle
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import nltk

# Download nltk libraries if necessary
nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])


def load_data(database_filepath):
    """
    Connect to database, retrieve data set, and split into X and Y sets
    :param database_filepath:
    :return:
    """
    # Connect to database
    engine = create_engine(database_filepath)

    # Read messages table to dataframe
    df = pd.read_sql_table("messages", engine)

    # Set message column as X feature set
    X = df["message"]

    # Isolate category columns
    cat_cols = [
        col for col in df.columns if col not in ["id", "message", "original", "genre"]
    ]

    # Set category columns as Y target set
    Y = df[cat_cols]

    return X, Y, cat_cols


def tokenize(text):
    """
    Tokenize, lemmatize, and clean message strings
    :param text:
    :return:
    """
    # Tokenize text string
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and clean tokens
    clean_tokens = []
    for t in tokens:
        clean_token = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    Build model pipeline with transformation and evaluation steps
    :return:
    """

    # Create model pipeline
    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    tokenizer=tokenize,
                    max_df=1.0,
                    max_features=None,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("tfidf", TfidfTransformer(norm="l2", use_idf=False)),
            (
                "clf",
                MultiOutputClassifier(estimator=MultinomialNB(alpha=0.01), n_jobs=-1),
            ),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model using test splits.  Print classification report for each category.
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    # Generate predictions
    y_pred = model.predict(X_test)

    # Evaluate each category
    for i, c in enumerate(category_names):
        print("***", c, "***")
        print(classification_report(Y_test[c].values, y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save model as pickle file
    :param model:
    :param model_filepath:
    :return:
    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
