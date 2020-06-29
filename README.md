# Problem Statement
Build a unique model to extract text from images and classify them on the basis of excessive content (Ex: toxic, abusive, hatred, racist, sexist)
 
## Project Overview:  
* Using CV2 and Pytesseract to extract Textual information from screenshots
* Using Toxic Comment Classifier Data as a training dataset
* Selecting the `severe_toxic` table as the `dependent variable` and `comment_text` as the `independent variable`
* Data Preparation - Handling the slang text and useless characters
* Processing the data using NLTK
* Creating a model using Sklearn
* Pickle the model
* Creating a Flask API Endpoint to classify comments using the pretrained model

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** CV2, tesseract, pandas, numpy, sklearn, matplotlib, flask, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```   
**Flask Productionization:** https://github.com/pallets/flask

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width=200>](https://scikit-learn.org/stable/) [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) 

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Tesseract_OCR_logo_%28Google%29.png" width=270>](https://tesseract-ocr.github.io/) 

## Project Deployment Walk-Through
https://drive.google.com/file/d/15eZIfKGq7xCDytilxT9s5C8gsWqLFUpU/view?usp=sharing
![](https://github.com/mihir1493/Toxic-Comment-Classifier-for-Instagram/blob/master/demo_video.gif)

## EDA
The data is highly skewed as there are far less 'severe_toxic' classes as compared to the entire dataset. Thus downsampling has to be performed, causing loss of valuable information.
Secondly, the data is too much to be viable in processing on my local machine, hence I have picked a subset of data.

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Multinominal Naive Bayes** : Accuracy Score = 0.936
*   **Random Forest Classifier**: Accuracy Score = 0.924
*	**Logistic Regression**: Accuracy Score = 0.92
*	**Gradient Boosting Classifier**: Accuracy Score = 0.912

![alt text](https://github.com/mihir1493/Toxic-Comment-Classifier-for-Instagram/blob/master/results.JPG "Results")

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver. The comments were then passed to a CountVectorizer to tokenize the text and parsed thought the model to evaluate the result. 

## Team
[![Devesh](https://avatars2.githubusercontent.com/u/49936431?s=200&v=4)](https://github.com/deveshdatwani) 
[![Mihir](https://avatars3.githubusercontent.com/u/56906607?s=200&v=4)](https://github.com/mihir1493)


