# ML Vocab 
> covering AI ML DeepLearning and terms an AI/ML engineer would need to know.

## Table of Contents 

1. [Artificial Intelligence](#artificial-intelligence)  
1. [Machine Learning](#machine-learning)  
1. [Algorithm](#algorithm)  
1. [Data](#data)  
1. [Model](#model)  
1. Model Fitting  
1. Training Data  
1. Test Data  
1. Supervised Learning  
1. Unsupervised Learning  
1. Reinforcement Learning  
1. Feature (Input, Independent Variable, Predictor)  
1. Feature Engineering  
1. Feature Scaling  
1. Dimensionality  
1. Target  
1. Instance  
1. Lable  
1. Model Complexity  
1. Bias & Vairance  
1. Bias Variance Tradeoff  
1. Noise  
1. Overfitting & Underfitting  
1. Validation & Cross Validation  
1. Regularization  
1. Batch, Epoch, Iteration  
1. Parameter  
1. Hyperparameter  
1. Cost Function (Loss Function, Objective Function)  
1. Gradient Descent  
1. Learning Rate   
1. Evaluation  

---

## Artificial Intelligence

The capability of machines to perform tasks that typically require human intelligence  
Can include language, recognizing images, solving problems, making decisions 

AI aims to mimic human cognitive function through techniques like machine learning but not all AI is machine learning for example chess playing engines are not ML because they are a predefined set of search algorithms to follow

Examples of machine learning: Automatic Translation, Traffic Prediction, Virutal Personal Assistant, Image Recognition, Email Spam Filtering, Text and Speech Recognition, Search Engine Results, Online Fraud Detection, Medical Diagnosis

## Machine Learning 

Allows computers to learn from data to improve performance on tasks over time without being explicitly programmed for each task  

Ex. Spam Email Detector is trained on examples of both spam and non-spam emails learning wrods, phrases, or patterns typically found in spam-messages. overtime the model will get better at predicting accurately.

This simulates a human child seeing something many times and recognizing the patterns over time. Like telling the difference between a cat and a dog you learn from the animal being pointed out by others.

## Algorithm 

Well defined set of instructions or rules that a computer follows to solve a problem or perform a task.  

Literally everything even sorting is an algorithm, encryption.  

Step by step recipe to success is an algorithm.  

## Data

Information that can be collected, analyzed, and used to make decisions predictions or provide insights  
Usually Numbers, Text, or Images

## Model 

Mathematical representation trained to recognize patterns  
Most common type is simply a mapping function between an input and output  
In linear regression and is simply the equation of the final regression line. for example a model predicts linear relationship between square footage of a house and its price.  

The fitting of the data on the line becomes the trained model  
The tained model is the intersection and slope of the line 

## Model Fitting 

Also called training or learning.  

Adjust parameters to find best match between model prediction and the actual data.  

Think with linear regression, keep trying different lines till you find the best fit

## Training Data

carefully selected subset of data to teach machine learning models how to make predictions.  
Input examples paired with their correct outputs

## Test Data 

separate collection of data used to evaluate how well a machine learning model performs on examples it hasn't seen  
helps to verify if it can actually make good predictions

Test and Training data are separated randomly before beginning the modeling process so that the model can never see the test data before running final test.

Any inclusion of test data in training data is called Data Leakage

## Supervised Learning 

Foundational approch where models learn from labeled examples meaning the true outcomes or targets are known  
Think a study guide with the answers.

70% of ML

## Unsupervised Learning

Models learn to find patterns and structure in data without being given labeled examples or correct answers.  

rather than being taught these algorithms discover natural groupings and relationships within the data on their own.

helps uncover hidden patterns.  

## Reinforcement Learning

newer branch in late 2010s  
learns from interaction and feedback  
learns through trail and error getting rewarded for good decisions and penalized for poor ones

## Feature (Input, Independent Variable, Predictor)

specifice piece of input, literally the actual number in the data

## Feature Engineering

process of creating new more informative features from existing data to improve a models performance  

aka data cleaned, structured, organized and more insight  
ex. date -> day_of_week, is_holiday  
will help sales much better than just date

the difference between average and excellent

## Feature Scaling

Normalization or Standardization  
transforming numerical features to similar scale to prevent larger ranges from dominating the learning process
ex. salary vs age  

Xnorm = (x-min(x)) / (max(x)-min(x))  
ex. age=35, min=27, max=48; Xnorm = (35-27) / (48-27) = 8 / 21 = 0.3809  

Xstand = (x-mean(x)) / (standard deviation(x))  
think scaling in range 0-1

## Dimensionality

number of features, dimentions, or attributes.  
each column in an excel spreadsheet of data is considered a dimension  

Dimensionality reduction techniques are cruitial to compress many features into a smaller set while preserving important information  

## Target 

Output variable, what the model is trying to predict based upon the features.  
basically in house model it would have target as price, spam filter would have target as spam.  

## Instance 

sample of data, a single complete unit of data that includes all features and in supervised learning its target value

## Label  

class, target value, the label is the answer so the name cat under a cat is the label for the right answer

## Model Complexity 

how complex, its ability to capture patterns in data  
often more complex model has more parameters and can learn more complicated relationships

finding the right amount of complexity is important, too complex and it can be overfitted, too simple and it can be underfitting  

## Bias & Vairance 

how limited or inflexible a models assumptions are about the underlying patterns in the data

a linear regression can have a high bias since it makes simple assumptions

a good balance is low bias and low variance

## Bias Variance Tradeoff 

fundamental concept. models ability to minimize bias and variance at the same time  
as complexity increases bias decreases because the model can capture more complex patterns but variance increases because of increased sensitivty in to the training data

*Dive Deeper* -> [coming-soon](/AI_ML_Knowledge/BV_Tradeoff.md)

## Noise

random variation or errors in data, we want to ignore the noise after fitting the data

## Overfitting & Underfitting

Overfitting is when the model learns the noise and random data rather than learning the true underlying patterns  
like memorizing answers without memorizing the concepts

Underfitting is when the model is too simple to capture the underlying patterns in the data

## Validation & Cross Validation 

practice of evaluating a models performance on data it hasnt been trained on by setting aside a portion of the training data called the validation set to simulate how well the model will perform on new unseen data.

cross validation extends this by repeatedly training and validating the model on different splits of data

## Regularization 

technique used to prevent overfitting by adding constraints or penalties that discorage a model from becoming too complex or fitting too closely to the training data.

strength of regularization is a hyperparameter  

too much regularization leads to underfitting  

## Batch, Epoch, Iteration

Batch is a subset of training data that is processed together in a single step of model training rather than processing the entire dataset at once.  
Batch size is a hyper parameter.  
Larger batches provide more stable parameter updates. take more memory  
Smaller batches update more frequently allowing model to escape local optima

> the hellie is local optima

Epoch is complete pass through the entire training set during model training. Too many epochs cause overfitting

Iteration is the number of batches in an epoch

## Parameter

Model parameter or weight, learns during training from the data unlike hyperparameters which are set before training begins  
Finding parameters of a model is the goal of the training process

in linear regression model slope M and intercept B are parameters that the model adjusts to fit the data 

more complex models parameters include all weights and biases that are automatically adjusted during training to minimize errors

some models have billions of parameters each fine tuned through the training process.

## Hyperparameter 

config setting used to control the learning process before training begins

## Cost Function (Loss Function, Objective Function)

level of how wrong a model is he difference between predicted and real

## Gradient Descent

optimization alg used to train models by iteravely adjusting parameters to minimize errors 

## Learning Rate 

how much a model can adjust its parameters during training

## Evaluation

how well it performs with data never seen before