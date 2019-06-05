# Disaster Response Message Classification

## Overview

This repository supports a web application for classifying text messages into categories related to disaster relief efforts.

[View the application here.](https://disaster-relief-zg.herokuapp.com/)

## Structure

```
- app
| - template
| |- master.html            # Main page of web app
| |- go.html                # Classification result page of web app
|- run.py                   # Flask file that runs app

- data
|- disaster_categories.csv  # Data to process 
|- disaster_messages.csv    # Data to process
|- process_data.py          # Python script for parsing data set
|- DisasterRelief.db        # Database to save clean data to

- models
|- train_classifier.py      # Python script for training classification model
|- model.pkl                # Model pickle file

- README.md
- Profile
- requirements.txt
```


## Modeling

Natural language processing techniques were used to generate features from the message text. Classification was performed using a multi-output [multinomial naive Baye's classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## Web Application

Web application deployed to Heroku using Flask framework.

## Executing Locally
Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database 
```python
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

- To run ML pipeline that trains classifier and saves
```python
python models/train_classifier.py data/DisasterResponse.db models/model.pkl
```
- Run the following command in the app's directory to run your web app. 
```python
python run.py
```

## Acknowledgements

This repository is related to a course project from the Udacity Data Science Nanodegree.  Special thanks to [Udacity](https://www.udacity.com) and [Figure Eight](https://www.figure-eight.com/) for providing the inspiration and data set for this project.

## Citations
Li, Susan, and Susan Li. “Multi-Class Text Classification with Scikit-Learn.” Towards Data Science, Towards Data Science, 19 Feb. 2018, towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f.