
# Stack Overflow Tag Prediction using Logistic Regression

## I. Definition
### Project Overview
Stack Overflow is the largest, most trusted online community for developers to learn, share their programming knowledge, and build their careers.

Stack Overflow is something which every programmer use one way or another. Each month, over 50 million developers come to Stack Overflow to learn, share their knowledge, and build their careers. It features questions and answers on a wide range of topics in computer programming. The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. As of April 2014 Stack Overflow has over 4,000,000 registered users, and it exceeded 10,000,000 questions in late August 2015. Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: Java, JavaScript, C#, PHP, Android, jQuery, Python and HTML.

### Problem Statement
Suggest the tags based on the content that was present in the question posted on Stackoverflow.

### Metrics
In order to evaluate the model, we will use the micro F1 score to test accuracy.
Micro-Averaged F1-Score (Mean F Score) : The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

```math
F1 = 2 * precision*recall / (precision + recall)
```
In the multi-class and multi-label case, this is the weighted average of the F1 score of each class. 
'Micro f1 score': 
Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance. 

## II. Analysis

### Data Exploration

The dataset contains 6,034,195 rows. The columns in the table are:
  * Id - Unique identifier for each question
  * Title - The question's title
  * Body - The body of the question
  * Tags - The tags associated with the question in a space-seperated format (all lowercase, should not contain tabs '\t' or ampersands     '&')
<p align="center">
<img src="Images/Dataset.PNG" width="800" height="250" />
</p>

Tags are predicted using Body of the question and Title which is an unstructured data. We can Identify the question using the Unique Id feature. 
As working with such a huge data set entails many computational limitations, so decided to use only a subset of the data.
There are 42048 number of unique tags. ".a, .app, .asp.net-mvc, .aspxauth, .bash-profile, .class-file, .cs-file, .doc, .drv, .ds-store" are some of the tags. By seeing the tags we can say the questions are all related to computer programming. 

Below is the distribution of tags

<p align="center">
<img src="Images/Tags.PNG" width="700" height="200" />
</p>

<p align="center">
<img src="Images/Frequent_tags.PNG" width="700" height="200" />
</p>

  * Majority of the most frequent tags are programming language.
  * C# is the top most frequent programming language.
  * Android, IOS, Linux and windows are among the top most frequent operating systems.
