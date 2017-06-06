
The _Housing_ data set is a popular regression benchmarking data set hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). It contains 506 records consisting of multivariate data attributes for various real estate zones and their housing price indices. The task is then to learn a regression model that can predict the price index or range.

## Attribute Information:

1. **CRIM**: per capita crime rate by town
2. **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS**: proportion of non-retail business acres per town
4. **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. **NOX**: nitric oxides concentration (parts per 10 million)
6. **RM**: average number of rooms per dwelling
7. **AGE**: proportion of owner-occupied units built prior to 1940
8. **DIS**: weighted distances to five Boston employment centres
9. **RAD**: index of accessibility to radial highways
10. **TAX**: full-value property-tax rate per $10,000
11. **PTRATIO**: pupil-teacher ratio by town
12. **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13. **LSTAT**: % lower status of the population
14. **MEDV**: Median value of owner-occupied homes in $1000's

## Model

Below is a GP model for predicting the **MEDV**

$$
	\begin{align}
		& MEDV(\mathbf{u}) = f(\mathbf{u}) + \epsilon(\mathbf{u}) \\
		& f \sim \mathcal{GP}(m(\mathbf{u}), K(\mathbf{u},\mathbf{v})) \\
		& \mathbb{E}[\epsilon(\mathbf{u}).\epsilon(\mathbf{v})] = K_{noise}(\mathbf{u}, \mathbf{v})\\
	\end{align}
$$

## Syntax

The [```TestGPHousing()```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-examples/index.html#io.github.mandar2812.dynaml.examples.TestGPHousing$) program can be run in the REPL, below is a description of each of its arguments.

Parameter | Type | Default value |Notes
--------|-----------|-----------|------------|
kernel | [`#!scala CovarianceFunction`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.CovarianceFunction) | - | The kernel function driving the GP model.
noise | [`#!scala CovarianceFunction`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.CovarianceFunction) | - | The additive noise that corrupts the values of the latent function.
trainFraction | `#!scala Double` | 0.75 | Fraction of the data to be used for model training and hyper-parameter selection.
columns | `#!scala List[Int]` | 13, 0,.., 12 | The columns to be selected for analysis (indexed from 0), first one is the target column.
grid| `#!scala Int` | 5 | The number of grid points for each hyper-parameter  
step | `#!scala Double`| 0.2| The space between grid points.
globalOpt | `#!scala String` | ML | The model selection procedure `#!scala "GS", "CSA",` or `#!scala "ML"`
stepSize | `#!scala Double` | 0.01 | Only relevant if `#!scala globalOpt = "ML"`, determines step size of steepest ascent.
maxIt | `#!scala Int` | 300 | Maximum iterations for ML model selection procedure.


```scala
TestGPHousing(
  kernel = new FBMKernel(0.55) + new LaplacianKernel(2.5),
  noise = new RBFKernel(1.5),
  grid = 5, step = 0.03,
  globalOpt = "GS", trainFraction = 0.45)
```

```
16/03/03 20:45:41 INFO GridSearch: Optimum value of energy is: 278.1603309851301
Configuration: Map(hurst -> 0.4, beta -> 2.35, bandwidth -> 1.35)
16/03/03 20:45:41 INFO SVMKernel$: Constructing kernel matrix.
```

```
16/03/03 20:45:42 INFO GPRegression: Generating error bars
16/03/03 20:45:42 INFO RegressionMetrics: Regression Model Performance: MEDV
16/03/03 20:45:42 INFO RegressionMetrics: ============================
16/03/03 20:45:42 INFO RegressionMetrics: MAE: 5.800070254265218
16/03/03 20:45:42 INFO RegressionMetrics: RMSE: 7.739266267762397
16/03/03 20:45:42 INFO RegressionMetrics: RMSLE: 0.4150438478412412
16/03/03 20:45:42 INFO RegressionMetrics: R^2: 0.3609909626630624
16/03/03 20:45:42 INFO RegressionMetrics: Corr. Coefficient: 0.7633838930006132
16/03/03 20:45:42 INFO RegressionMetrics: Model Yield: 0.7341944950376289
16/03/03 20:45:42 INFO RegressionMetrics: Std Dev of Residuals: 6.287519509352036
```

## Source Code

Below is the example program as a github gist, to view the original program in DynaML, click [here](https://github.com/transcendent-ai-labs/DynaML/blob/master/src/main/scala/io/github/mandar2812/dynaml/examples/TestGPHousing.scala).

<script src="https://gist.github.com/mandar2812/bc5ff898ca921f22b5ee.js"></script>
