This submission contains the code and report for our Twitter Sentiment Analysis project towards CS4225/CS5425.

For code submission, this directory contains all the scripts from our project, as described below:

1. Preprocessor.scala - This script takes in the raw dataset and preprocesses it into a clean set of tweets for the Machine Learning step, using regular expressions to identify various text patterns that need to be removed.
2. MachineLearning.scala - All the code for training the Deep Neural Network model and performing Grid Search and K-fold Cross Validation, using Spark MLLib.
3. EndToEnd.scala - This code uses the pretrained models for each sentiment, to predict sentiments for a set of tweets loaded from a CSV file (which contains tweets for a given hashtag), and it also writes the results of Sentiment Analysis and finding the top 150 words to an Azure SQL database using Spark SQL.
4. Utility.scala - Contains various constants used in all the scripts above.
5. build.sbt - SBT Build file to help import all the necessary Spark libraries to run the code.