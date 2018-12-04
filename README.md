# Stack Overflow Tag Prediction using Logistic Regression

## Project Overview

Stack Overflow is the largest, most trusted online community for developers to learn, share their programming knowledge, and build their careers.

Stack Overflow is something which every programmer use one way or another. Each month, over 50 million developers come to Stack Overflow to learn, share their knowledge, and build their careers. It features questions and answers on a wide range of topics in computer programming. The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. As of April 2014 Stack Overflow has over 4,000,000 registered users, and it exceeded 10,000,000 questions in late August 2015. Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: Java, JavaScript, C#, PHP, Android, jQuery, Python and HTML.

## Install
This project requires Python 3.5 and the following Python libraries installed:

  * NumPy
  * Pandas
  * matplotlib
  * scikit-learn
  * tensorflow
  * seaborn
  * string
  * collections
  * wordcloud
You will also need to have software installed to run and execute an iPython Notebook

We recommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

## Data Source 
https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data 

## Code
The code is provided in the SO_Tag_Predictor.ipynb notebook file. 

## Data
  * Data is divided into 2 files: Train and Test.
  * Train.csv contains 4 columns: Id,Title,Body,Tags.
  * Test.csv contains the same columns but without the Tags, which you are to predict.
  * Size of Train.csv - 6.75GB
  * Size of Test.csv - 2GB
  * Number of rows in Train.csv = 6034195

**Datafields**
 * Dataset contains 6,034,195 rows.

**Features**
  * Id - Unique identifier for each question
  * Title - The question's title
  * Body - The body of the question

**Target Variable**
  * Tags - The tags associated with the question in a space-seperated format (all lowercase, should not contain tabs '\t' or ampersands '&')
  

  
