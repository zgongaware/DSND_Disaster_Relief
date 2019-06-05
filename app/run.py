import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


def plot_category_occurrences(df):
    """
    Plot occurrence of categories in the data set
    :param df:
    :return:
    """

    # Isolate category columns
    cat_cols = [
        col for col in df.columns if col not in ["id", "message", "original", "genre"]
    ]

    # Retrieve sums of category occurrences
    dist = df[cat_cols].sum().sort_values(ascending=False)

    # Format category names
    names = dist.index.str.title().str.replace("_", " ").to_list()

    # Get list of color
    colors = ["#E1396C" for i in range(len(dist))]

    # Define graph
    graph = {
        "data": [
            Bar(
                x=names,
                y=dist.values.tolist(),
                opacity=0.7,
                marker={"color": colors, "line": {"color": colors, "width": 1.5}},
            )
        ],
        "layout": {
            "title": "Occurrence of Message Categories",
            "xaxis": {"title": "Category", "automargin": True},
            "yaxis": {"title": "Count", "automargin": True},
            "autosize": True,
        },
    }

    return graph


def plot_message_genres(df):
    """
    Plot message genre distribution
    :param df:
    :return:
    """
    # Aggregate genre counts
    genre_counts = df.groupby("genre").count()["message"]

    # Format names
    genre_names = genre_counts.index.str.title().to_list()

    # Define graph
    graph = {
        "data": [
            Pie(
                labels=genre_names,
                values=genre_counts,
                hoverinfo="label+percent",
                textinfo="value",
                textfont={"size": 20},
                marker={
                    "colors": ["#FEBFB3", "#E1396C", "#96D38C"],
                    "line": {"color": "#000000", "width": 1},
                },
            )
        ],
        "layout": {"title": "Distribution of Message Genres"},
    }

    return graph


# Load data
engine = create_engine("sqlite:///data/DisasterRelief.db")
df = pd.read_sql_table("messages", engine)

# Load model
model = joblib.load("models/model.pkl")

# Index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    graphs = list()
    graphs.append(plot_message_genres(df))
    graphs.append(plot_category_occurrences(df))

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route("/go")
def go():
    # Save user input in query
    query = request.args.get("query", "")

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run()


if __name__ == "__main__":
    main()
