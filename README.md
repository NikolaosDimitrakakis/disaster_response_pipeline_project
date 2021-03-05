# Disaster Response Pipeline Project
A Dashboard to classify messages related to disasters

### Table of Contents

 1. [Installation](#installation)
 2. [Project Motivation](#motivation)
 3. [File Descriptions](#files)
 4. [Execution](#execution)
 5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

For the project we used Machine Learning libraries(NumPy, SciPy, Pandas, Sciki-Learn), Natural Language Process Libraries(NLTK), SQLlite Database Libraqries(SQLalchemy), Model Loading and Saving Library(Pickle), Web App and Data Visualization(Flask, Plotly).  The code should run with no issues using   Python versions 3.*.

## Project Motivation<a name="motivation"></a>

The project scope is to build a Natural Language Processing Model to categorize messages on real time basis. The dataset contains pre-labelled tweets and messages from real-life disaster events and is preprocessed by Figure Eight.

## File Descriptions <a name="files"></a>

There are 3 files available here with work related to the project.  Each of the files is a key to run the dashboard.  Markdown cells were used to assist in walking through the thought process for individual steps.
In the app file are all the needed templates to run a web app to show us the model results in real time.
In the data file we have all the information for processing the data, building the ETL pipeline to extract the data, clean and save them in SQLite DB.
In the model file we build the machine learning pipeline to load the data, train a model, and save the trained model as .pkl file.
There are two notebooks where we created the ETL pipeline and the model we used for the app.

## Execution<a name="execution"></a>

1. You can run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for providing the data. Also Udacity for giving me the chance to learn Data Science in depth. Otherwise, feel free to use the code here as you would like! 
