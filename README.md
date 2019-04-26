# Credit Card Transaction Fraud Detection System

## *Introduction*

This project is my individual EE551 Python project for Spring 2019 term.

## *Proposal*

This system will help in detecting fraud/fake credit card transactions using Naive Bayes classifier.

## *Features*

- The system will analyze a data which will consist of hundreds of thousands of credit card transactions.
- Based on the analysis, the system will be able to detect fraud/fake transactions.

## *To-Do*

- Import the data using pandas.
- Convert all the input features into some other features using PCA.
- Plot graph of fraud transactions and genuine transactions.
- Train the classifier to detect fraud points from the graphs.

## *Project Explanation*

- Firstly, I imported all the necessary libraries.
- Secondly, I imported the dataset from the creditcard.csv file using pandas dataframe.
- Then, I visualised the data and found that the data was highly unbalanced because there were less than 1% of fraud transactions in the whole dataset.
- Then, I plot the KDE (Kernel Density Estimate) for every feature of the dataset.
- Then, using the sklearn.preprocessing package, I classified the dataset into training dataset and testing dataset.
- After training the system with the dataset, I found it that it was not very accurate.
- So, I had to remove some of the features from the dataset which were not helping the system in distinguishing between Fair and Fraud transactions.
- Again training the system with the new dataset, I found the system to be more accurate than with the previous dataset.

## *Author*

Abhishek Sangani

## *License*

This project is licensed under the GNU General Public License v3.0. For more information, see the [LICENSE](LICENSE.md) file for details.
