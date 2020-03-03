## Supervised machine learning

**What is supervised machine learning? 👶**

A case when we have both features (the matrix X) and the labels (the vector y)

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

Regression is a part of supervised ML. Regression models predict a real number

**What is linear regression? When do we use it? 👶**

Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y).

With a simple equation:

```
y = B0 + B1*x1 + ... + Bn * xN
```

B is regression coefficients, x values are the independent (explanatory) variables  and y is dependent variable.

The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

Simple linear regression:

```
y = B0 + B1*x1
```

Multiple linear regression:

```
y = B0 + B1*x1 + ... + Bn * xN
```

<br/>

**What’s the normal distribution? Why do we care about it? 👶**

Answer here

<br/>

**How do we check if a variable follows the normal distribution? ‍⭐️**

Answer here

<br/>

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

Answer here

<br/>

**What are the methods for solving linear regression do you know? ‍⭐️**

Answer here

<br/>

**What is gradient descent? How does it work? ‍⭐️**

Answer here

<br/>

**What is the normal equation? ‍⭐️**

Normal equations are equations obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares); normal equations allow one to estimate the parameters of a multiple linear regression.

<br/>

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

Answer here

<br/>

**Which metrics for evaluating regression models do you know? 👶**

The mean square error, root mean square error, median absolute deviation, normalized median absolute deviation, etc.
<br/>

**What are MSE and RMSE? 👶**

Answer here

<br/>


## Validation

**What is overfitting? 👶**

Overfitting is when a model is tuned to perform very well on the training data set but cannot generalize to the test data set. It has low bias and high variance.
<br/>

**How to validate your models? 👶**

The first step is to split the full data set into a training or testing subset. This ensures that the model is only trained on a (representative) portion of the data, and provides us with data with which to compare the predictions of our model.
<br/>

**Why do we need to split our data into three parts: train, validation, and test? 👶**

* Train -- This is the data set used to train the model.
* Validation -- This portion of the data is used to initially evaluate the model's performance and to perform any hyperparameter tuning or other optimizations.
* Test -- This is the final hold out set of data that the model has never seen before.

It is important to have *three* parts of the data to ensure that the model is the as flexible as it can be. The validation portion allows us to tune the model, but still have a portion of the data that is useful for evaluating the final performance.
<br/>

**Can you explain how cross-validation works? 👶**

Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting different training and validation sets, in order to reduce the bias that you would have by selecting only one validation set.
<br/>

**What is K-fold cross-validation? 👶**

K-fold cross validation is when we "fold" the training data K times, generating K training-validation sets.
<br/>

**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

It largely depends on how many training examples we have. If there are only 10 training examples, then *k=10* wouldn't make much sense, as there is only a single training example in each case.

My favorite is *k = 5* because that is roughly an 80-20 split between training and validation.
<br/>


## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Classification is labeling a set of observations into two or more categories. You can use many of the same types of models for classification that you could use for regression.
<br/>

**What is logistic regression? When do we need to use it? 👶**

logistic regression is a linear model where the predicted value is either 0 or 1. See this: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
<br/>

**Is logistic regression a linear model? Why? 👶**

Yes. [This](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model?share=1) seems like a good answer.

From that link:

> Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. Or in other words, the output cannot depend on the product (or quotient, etc.) of its parameters!

Here's a bit more: Logistic regression produces a linear decision boundary because the additivity of the terms: Our outcome z depends on the additivity of the parameters, e.g., :
```
z = w_1 * x_1 + w_2 * x_2
```
There's no interaction between the parameter weights, nothing like w_1*x_1 * w_2* x_2 or so, which would make our model non-linear!

<br/>

**What is the sigmoid function? What does it do? 👶**

Answer here

<br/>

**How do we evaluate classification models? 👶**

We compare the predicted classes to the true classes. In general, the more observations where we correctly predict the class the better the model is.
<br/>

**What is accuracy? 👶**

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

<br/>

**Is accuracy always a good metric? 👶**

Accuracy is not a good performance metric when there is imbalance in the dataset. For example, in binary classification with 95% of A class and 5% of B class, prediction accuracy can be 95%. In case of imbalance dataset, we need to choose Precision, recall, or F1 Score depending on the problem we are trying to solve.

<br/>

**What is the confusion table? What are the cells in this table? 👶**

Confusion table (or confusion matrix) shows how many True positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN) model has made.

||                |     Actual   |        Actual |
|:---:|   :---:        |     :---:    |:---:          |
||                | Positive (1) | Negative (0)  |
|**Predicted**|   Positive (1) | TP           | FP            |
|**Predicted**|   Negative (0) | FN           | TN            |

* True Positives (TP): When the actual class of the observation is 1 (True) and the prediction is 1 (True)
* True Negative (TN): When the actual class of the observation is 0 (False) and the prediction is 0 (False)
* False Positive (FP): When the actual class of the observation is 0 (False) and the prediction is 1 (True)
* False Negative (FN): When the actual class of the observation is 1 (True) and the prediction is 0 (False)

Most of the performance metrics for classification models are based on the values of the confusion matrix.

<br/>

**What are precision, recall, and F1-score? 👶**

* Precision (P) and Recall (R) are classification evaluation metrics:
* P = TP / (TP + FP) and R = TP / (TP + FN).
* Where TP is true positives, FP is false positives and FN is false negatives
* In both cases the score of 1 is the best: we get no false positives or false negatives and only true positives.
* F1 is a combination of both precision and recall in one score:
* F1 = 2 * PR / (P + R).
* Max F score is 1 and min is 0, with 1 being the best.

<br/>

**Precision-recall trade-off ‍⭐️**

For a given model, we can either increase the precision or the recall at the expense of the other. If we want to accurately predict a few high-risk situations we would want higher recall, because it might be better to treat everything upfront. If we need to identify more lower-risk situations then precision might be better because the cost of missing some is lower.
<br/>

**What is the ROC curve? When to use it? ‍⭐️**

The Receiver Operating Characteristics (ROC) curve is a performance metric for classifications are various threshold settings. It is plotted as the *false positive rate* vs. *true positive rate* (recall) where the diagonal line represents 50-50 chance.
<br/>

**What is AUC (AU ROC)? When to use it? ‍⭐️**

The area under the curve (AUC) represents a model's ability to discriminate between classes correctly. The higher the AUC the better the model is at correctly predicting classes.

<br/>

**How to interpret the AU ROC score? ‍⭐️**

When the AUC is close to 1 the model is has a good measure of separability between the classes. When the value is close to 0, then the model is predicting exactly the opposite of what we want 1 when should be 0. A value close to 0.5 means that the model has very little power to separate the classes.
<br/>

**What is the PR (precision-recall) curve? ‍⭐️**

Shows the frontier boundary between the precision and recall for a given model. As you increase one the other will decrease, *for the given model.*

<br/>

**What is the area under the PR curve? Is it a useful metric? ‍⭐️I**

Answer here

<br/>

**In which cases AU PR is better than AU ROC? ‍⭐️**

Answer here

<br/>

**What do we do with categorical variables? ‍⭐️**

We either use a model that can handel directly (random forest?) or we transform them into a numerical feature that we can feed into a different model.
<br/>

**Why do we need one-hot encoding? ‍⭐️**

Because encoding categorical variables with a simple linear scale `[1,2,3,..]` can imply different weights to the variables. We use one-hot encoding to convert categorical variables into number values without implicitly assigning them (some times misleading) weights.
<br/>


## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

Answer here

<br/>

**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

Answer here

<br/>

**What is regularization? Why do we need it? 👶**

Regularization is a tool to help prevent overfitting of our model. It adds a penalty term that increases as model complexity increases.
<br/>

**Which regularization techniques do you know? ‍⭐️**

L1, L2
<br/>

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

Answer here

<br/>

**How does L2 regularization look like in a linear model? ‍⭐️**

L2 regularization penalizes the model with the square of the weights
<br/>

**How do we select the right regularization parameters? 👶**

Cross validation?
<br/>

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

It reduces them to small, but non-zero values.
<br/>

**How L1 regularization looks like in a linear model? ‍⭐️**

L1 penalizes the model with the absolute value of the weights.

<br/>

**What’s the difference between L2 and L1 regularization? ‍⭐️**

The penality term. A regression model that uses L1 regularization technique is called *Lasso Regression* and model which uses L2 is called *Ridge Regression*.

* L1 - Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

* L2 - Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.

<br/>

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Yes, that is called elastic net.
<br/>

**What’s the interpretation of the bias term in linear models? ‍⭐️**

Answer here

<br/>

**How do we interpret weights in linear models? ‍⭐️**

If the variables are normalized, we can interpret weights in linear models like the importance of this variable in the predicted result.

<br/>

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

Yes, but only for the case above.
<br/>

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

Answer here

<br/>


## Feature selection

**What is feature selection? Why do we need it? 👶**

It is selecting only features which significantly contribute to the model. We need to do it because if we are including many low-signal features, then we are just adding noise and confusion to the model and not actually increasing its predictive power (or maybe only slightly).
<br/>

**Is feature selection important for linear models? ‍⭐️**

Yes. For the same reasons as above, including the less-predictive features only adds noise to the model.
<br/>

**Which feature selection techniques do you know? ‍⭐️**

Answer here

<br/>

**Can we use L1 regularization for feature selection? ‍⭐️**

Yes!
<br/>

**Can we use L2 regularization for feature selection? ‍⭐️**

Not really.
<br/>


## Decision trees

**What are decision trees? 👶**

Answer here

<br/>

**How do we train decision trees? ‍⭐️**

We use the CART algorithm. At each decision point in the tree the dataset is split in order to either maximize the purity (classification) or the variance (regression). Repeat this process until the desired depth.

For classification trees, scikit-learn tries to minimize the Gini impurity.

```
G = sum(fi * (1-fi))
```
Summed from 1 to *C*, where *fi* is the frequency of the label *i* at a node and *C* is the number of unique labels (features).

For regression, we are going to use the variance reduction (in each split). The variance is calculated by the Mean Squared Error

```
MSE = 1/N * sum((yi - mu)**2)
```
where *yi* is the label (feature) for an instance, *N* is the total number of instances and *mu* is the mean value of all instances


<br/>

**What are the main parameters of the decision tree model? 👶**

The depth of the tree (the number of split nodes).

<br/>

**How do we handle categorical variables in decision trees? ‍⭐️**

Answer here

<br/>

**What are the benefits of a single decision tree compared to more complex models? ‍⭐️**

Answer here

<br/>

**How can we know which features are more important for the decision tree model? ‍⭐️**

From Toward Data Science:

>Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.

The importance of each node is given by:

```
nij = wj*Cj - wlefti*Cleftj - wrightj* Crightj
```
where
* nij = the importance of node j
* wj = weighted number of samples reaching node j
* Cj = the impurity value of node j
* leftj = child node from left split on node j
* rightj = child node from right split on node j

See [this](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3) Toward Data Science article for more equations.

<br/>


## Random forest

**What is random forest? 👶**

An ensemble method that uses a collection of decision trees to solve the regression or classification problem.
<br/>

**Why do we need randomization in random forest? ‍⭐️**

Training on a random subset of the data helps to prevent overfitting.
<br/>

**What are the main parameters of the random forest model? ‍⭐️**

Number of trees, depth, etc.
<br/>

**How do we select the depth of the trees in random forest? ‍⭐️**

Answer here

<br/>

**How do we know how many trees we need in random forest? ‍⭐️**

Answer here

<br/>

**Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️**

Answer here

<br/>

**What are the potential problems with many large trees? ‍⭐️**

Answer here

<br/>

**What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀**

Answer here

<br/>

**What happens when we have correlated features in our data? ‍⭐️**

Answer here

<br/>


## Gradient boosting

**What is gradient boosting trees? ‍⭐️**

Answer here

<br/>

**What’s the difference between random forest and gradient boosting? ‍⭐️**

Gradient boosted trees (GBT) differ from a random forest because in GBTs we are training one tree at a time, iteratively, one after another. In a random forest, we are using a random subset of the data and training all of the trees in parallel.
<br/>

**Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️**

Answer here

<br/>

**Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️**

Answer here

<br/>

**Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀**

Answer here

<br/>

**What are the main parameters in the gradient boosting model? ‍⭐️**

The learning rate and the number of trees.
<br/>

**How do you approach tuning parameters in XGBoost or LightGBM? 🚀**

Answer here

<br/>

**How do you select the number of trees in the gradient boosting model? ‍⭐️**

Answer here

<br/>



## Parameter tuning

**Which parameter tuning strategies (in general) do you know? ‍⭐️**

Grid and random search with cross validation.
<br/>

**What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️**

Grid search performs an exhaustive search of all possible parameters available to the model. A random search searches randomly through the parameter space, and then reports the best parameters. In principle you could use a random search to narrow the search space and then a grid search to refine the solution. In practice, I would use random search because it is faster than an exhaustive grid search.

<br/>


## Neural networks

**What kind of problems neural nets can solve? 👶**

Classification and regression.
<br/>

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

Answer here

<br/>

**Why do we need activation functions? 👶**

Answer here

<br/>

**What are the problems with sigmoid as an activation function? ‍⭐️**

Answer here

<br/>

**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

Answer here

<br/>

**How we can initialize the weights of a neural network? ‍⭐️**

Answer here

<br/>

**What if we set all the weights of a neural network to 0? ‍⭐️**

Answer here

<br/>

**What regularization techniques for neural nets do you know? ‍⭐️**

Answer here

<br/>

**What is dropout? Why is it useful? How does it work? ‍⭐️**

Answer here

<br/>


## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

Answer here

<br/>

**Which optimization techniques for training neural nets do you know? ‍⭐️**

Answer here

<br/>

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

Answer here

<br/>

**What’s the learning rate? 👶**

Answer here

<br/>

**What happens when the learning rate is too large? Too small? 👶**

Answer here

<br/>

**How to set the learning rate? ‍⭐️**

Answer here

<br/>

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

Answer here

<br/>

**When would you use Adam and when SGD? ‍⭐️**

Answer here

<br/>

**Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️**

Answer here

<br/>

**How do we decide when to stop training a neural net? 👶**

Answer here

<br/>

**What is model checkpointing? ‍⭐️**

Answer here

<br/>

**Can you tell us how you approach the model training process? ‍⭐️**

Answer here

<br/>


## Neural networks for computer vision

**How we can use neural nets for computer vision? ‍⭐️**

Answer here

<br/>

**What’s a convolutional layer? ‍⭐️**

Answer here

<br/>

**Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️**

Answer here

<br/>

**What’s pooling in CNN? Why do we need it? ‍⭐️**

Answer here

<br/>

**How does max pooling work? Are there other pooling techniques? ‍⭐️**

Answer here

<br/>

**Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀**

Answer here

<br/>

**What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️**

Answer here

<br/>

**What kind of CNN architectures for classification do you know? 🚀**

Answer here

<br/>

**What is transfer learning? How does it work? ‍⭐️**

Answer here

<br/>

**What is object detection? Do you know any architectures for that? 🚀**

Answer here

<br/>

**What is object segmentation? Do you know any architectures for that? 🚀**

Answer here

<br/>


## Text classification

**How can we use machine learning for text classification? ‍⭐️**

Answer here

<br/>

**What is bag of words? How we can use it for text classification? ‍⭐️**

Answer here

<br/>

**What are the advantages and disadvantages of bag of words? ‍⭐️**

Answer here

<br/>

**What are N-grams? How can we use them? ‍⭐️**

Answer here

<br/>

**How large should be N for our bag of words when using N-grams? ‍⭐️**

Answer here

<br/>

**What is TF-IDF? How is it useful for text classification? ‍⭐️**

Answer here

<br/>

**Which model would you use for text classification with bag of words features? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️**

Answer here

<br/>

**What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️**

Answer here

<br/>

**Do you know any other ways to get word embeddings? 🚀**

Answer here

<br/>

**If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️**

Answer here

<br/>

**How can you use neural nets for text classification? 🚀**

Answer here

<br/>

**How can we use CNN for text classification? 🚀**

Answer here

<br/>


## Clustering

**What is unsupervised learning? 👶**

Answer here

<br/>

**What is clustering? When do we need it? 👶**

Answer here

<br/>

**Do you know how K-means works? ‍⭐️**

Answer here

<br/>

**How to select K for K-means? ‍⭐️**

Answer here

<br/>

**What are the other clustering algorithms do you know? ‍⭐️**

Answer here

<br/>

**Do you know how DBScan works? ‍⭐️**

Answer here

<br/>

**When would you choose K-means and when DBScan? ‍⭐️**

Answer here

<br/>


## Dimensionality reduction
**What is the curse of dimensionality? Why do we care about it? ‍⭐️**

Answer here

<br/>

**Do you know any dimensionality reduction techniques? ‍⭐️**

PCA
<br/>

**What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️**

Answer here

<br/>


## Ranking and search

**What is the ranking problem? Which models can you use to solve them? ‍⭐️**

Answer here

<br/>

**What are good unsupervised baselines for text information retrieval? ‍⭐️**

Answer here

<br/>

**How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️**

Answer here

<br/>

**What is precision and recall at k? ‍⭐️**

Answer here

<br/>

**What is mean average precision at k? ‍⭐️**

Answer here

<br/>

**How can we use machine learning for search? ‍⭐️**

Answer here

<br/>

**How can we get training data for our ranking algorithms? ‍⭐️**

Answer here

<br/>

**Can we formulate the search problem as a classification problem? How? ‍⭐️**

Answer here

<br/>

**How can we use clicks data as the training data for ranking algorithms? 🚀**

Answer here

<br/>

**Do you know how to use gradient boosting trees for ranking? 🚀**

Answer here

<br/>

**How do you do an online evaluation of a new ranking algorithm? ‍⭐️**

Answer here

<br/>


## Recommender systems

**What is a recommender system? 👶**

Answer here

<br/>

**What are good baselines when building a recommender system? ‍⭐️**

Answer here

<br/>

**What is collaborative filtering? ‍⭐️**

Answer here

<br/>

**How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️**

Answer here

<br/>

**What is the cold start problem? ‍⭐️**

Answer here

<br/>

**Possible approaches to solving the cold start problem? ‍⭐️🚀**

Answer here

<br/>


## Time series

**What is a time series? 👶**

Answer here

<br/>

**How is time series different from the usual regression problem? 👶**

Answer here

<br/>

**Which models do you know for solving time series problems? ‍⭐️**

Answer here

<br/>

**If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️**

Answer here

<br/>

**You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**What are the problems with using trees for solving time series problems? ‍⭐️**

Answer here

<br/>
