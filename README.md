# Deployed_models
This repo contains deployed machine learning and deep learning models.


# Email/SMS Spam Classifier
The Email/SMS Spam classifier model takes an email or message as an input tokenize it, converts it into the vector form by the process of vectorization and using the vectors tries to predict and classifies whether the email or a sms is spam or not.
Here's a breakdown of how the model works:
1. *Data Collection:*  Training data is collected, which consists of emails and SMS messages that are already labeled as spam or not spam.

2. *Data Preprocessing:* The text data is cleaned and formatted for the machine learning model. This may involve removing punctuation, converting text to lowercase, and removing irrelevant words.

3. *Model Building*  A deep learning or a machine learning model is trained on the labeled data(in my case it's the deep learning model). As it is deep learning model hence a feedforward neural network is used here.

4. *Classification:* Once trained, the model can then analyze new emails and SMS messages and predict whether they are spam or not spam.
   
5. *Model Deployment:* Once satisfied with the model performance, deploy it in a production environment where it can make real-time classification. This could be in the form of a web application, API, or integration with other systems.
   
6. *Continuous Improvement:* Monitor the model's performance over time and update it periodically with new data to ensure accuracy and reliability. Consider retraining the model with updated datasets to incorporate any changes in data patterns.


![image](https://github.com/SP4567/Deployed_models/assets/92623123/9706b94b-c76f-4e37-99ab-4198441e4c71)



# Weather Sense
Weather sense is the deep learning model which takes various parametres as input such as Precipitation of a place, Minimum temperature, Maximum temperature, Wind Speed ans using these input it predicts there are chances of drizzle, rain, sun, snow, or fog.
Here's a breakdown of the model:

1. *Data Collection:* Gather historical weather data from reliable sources like government agencies or meteorological organizations. This data should include various features such as temperature, humidity, wind speed, precipitation, etc.
2. *Data Preprocessing:* Clean the collected data by handling missing values, outliers, and inconsistencies. Convert categorical variables into numerical representations if necessary. Split the data into training and testing sets.
3. *Feature Selection/Engineering:* Select relevant features that are most likely to influence weather predictions. You may also create new features based on domain knowledge or insights gained from the data.
4. *Model Selection:* Choose a suitable machine learning algorithm for weather prediction. Common choices include regression algorithms like linear regression, decision trees, random forests, or more advanced techniques like neural networks.
5. *Model Training:* Train your chosen model using the training dataset. Adjust hyperparameters as needed to optimize performance. Consider techniques like cross-validation to evaluate model performance and prevent overfitting.
6. *Model Evaluation:* Evaluate the trained model using the testing dataset. Use appropriate evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) to assess prediction accuracy.
7. *Model Deployment:* Once satisfied with the model performance, deploy it in a production environment where it can make real-time predictions. This could be in the form of a web application, API, or integration with other systems.
8. *Continuous Improvement:* Monitor the model's performance over time and update it periodically with new data to ensure accuracy and reliability. Consider retraining the model with updated datasets to incorporate any changes in weather patterns.
   

![image](https://github.com/SP4567/Deployed_models/assets/92623123/1597551e-2bb6-4452-beac-edd66b346bc9)





