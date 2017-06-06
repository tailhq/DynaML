---
title: "Wine Quality: Logit & Probit Models"
keywords: mydoc
tags: [examples, getting_started]
sidebar: product2_sidebar
permalink: p2_winequality_logistic.html
folder: product2
---

The [_wine quality_](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) data set is a common example used to benchmark classification models. Here we use the [DynaML](mandar2812.github.io/DynaML) scala machine learning environment to train classifiers to detect 'good' wine from 'bad' wine. A short listing of the data attributes/columns is given below. The UCI archive has two files in the wine quality data set namely ```winequality-red.csv``` and ```winequality-white.csv```. We train two separate classification models, one for red wine and one for white.

![Wine: Representative Image](/images/wine.jpg)

## Attribute Information:

### Inputs:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

### Output (based on sensory data):

12. quality (score between 0 and 10)

### Data Output Preprocessing

The wine quality target variable can take integer values from `0` to `10`, first we convert this into a binary class variable by setting the quality to be 'good'(encoded by the value `1`) if the numerical value is greater than `6` and 'bad' (encoded by value `0`) otherwise.


## Model

Below is a classification model for predicting the quality label $y$.


### Logit

$$
\begin{align}
  P(y \ = 1 \ | \ \mathbf{x}) &= \sigma(w^T \varphi(\mathbf{x}) + b) \\
  \sigma(z) &= \frac{1}{1 + exp(-z)}
\end{align}
$$

### Probit

The _probit regression_ model is an alternative to the _logit_ model it is represented as.

$$
\begin{align}
  P(y \ = 1 \ | \ \mathbf{x}) &= \Phi(w^T \varphi(\mathbf{x}) + b) \\
  \Phi(z) &= \int_{-\infty}^{z} \frac{1}{\sqrt{2 \pi}} exp(-\frac{z^{2}}{2}) dz  
\end{align}
$$



## Syntax

The [```TestLogisticWineQuality```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-examples/index.html#io.github.mandar2812.dynaml.examples.TestLogisticWineQuality$) program in the ```examples``` package trains and tests logit and probit models on the wine quality data.

Parameter | Type | Default value |Notes
--------|-----------|-----------|------------|
training | `#!scala Int` | 100 | Number of training samples
test | `#!scala Int` | 1000 | Number of test samples
columns | `#!scala List[Int]` | 11, 0, ... , 10 | The columns to be selected for analysis (indexed from 0), first one is the target column.
stepSize | `#!scala Double` | 0.01 | Step size chosen for [`#!scala GradientDescent`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GradientDescent)
maxIt | `#!scala Int` | 30 | Maximum number of iterations for gradient descent update.
mini | `#!scala Double` | 1.0 | Fraction of training samples to sample for each batch update.
regularization | `#!scala Double` | 0.5 | Regularization parameter.
wineType | `#!scala String` | red | The type of wine: red or white
modelType | `#!scala String` | logistic | The type of model: logistic or probit


## Red Wine


```scala
TestLogisticWineQuality(stepSize = 0.2, maxIt = 120,
mini = 1.0, training = 800,
test = 800, regularization = 0.2,
wineType = "red")
```

```
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Classification Model Performance
16/04/01 15:21:57 INFO BinaryClassificationMetrics: ============================
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Accuracy: 0.8475
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Area under ROC: 0.7968417788802267
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7493563745371187
```

![red-roc](/images/red-wine-logistic-roc.png)

![red-fmeasure](/images/red-wine-logistic-fmeasure.png)


## White Wine


```scala
TestLogisticWineQuality(stepSize = 0.26, maxIt = 300,
mini = 1.0, training = 3800,
test = 1000, regularization = 0.0,
wineType = "white")
```

```
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Classification Model Performance
16/04/01 15:27:17 INFO BinaryClassificationMetrics: ============================
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Accuracy: 0.829
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Area under ROC: 0.7184782682020251
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7182203962483446
```



![red-roc](/images/white-wine-logistic-roc.png)

![red-fmeasure](/images/white-wine-logistic-fmeasure.png)

## Source Code

<script src="https://gist.github.com/mandar2812/b309d5c26b5aba9c84415d2f7cd6d913.js"></script>
