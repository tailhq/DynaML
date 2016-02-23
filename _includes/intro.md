
The `data/` directory contains a few data sets, which are used by the programs in the `examples/` directory. Lets run a Gaussian Process (GP) regression model on the synthetic 'delve' data set.



```
DynaML>TestGPDelve("RBF", 2.0, 1.0, 500, 1000)
Feb 23, 2016 6:35:08 PM com.github.fommil.jni.JniLoader liberalLoad
INFO: successfully loaded /tmp/jniloader4173849050766409147netlib-native_system-linux-x86_64.so
16/02/23 18:35:09 INFO GPRegression: Generating predictions for test set
16/02/23 18:35:09 INFO GPRegression: Calculating posterior predictive distribution
16/02/23 18:35:09 INFO SVMKernel$: Constructing kernel matrix.
16/02/23 18:35:10 INFO SVMKernel$: Dimension: 500 x 500
16/02/23 18:35:10 INFO SVMKernel$: Constructing kernel matrix.
16/02/23 18:35:13 INFO SVMKernel$: Dimension: 1000 x 1000
16/02/23 18:35:13 INFO SVMKernel$: Constructing cross kernel matrix.
16/02/23 18:35:13 INFO SVMKernel$: Dimension: 500 x 1000
Feb 23, 2016 6:35:15 PM com.github.fommil.jni.JniLoader load
INFO: already loaded netlib-native_system-linux-x86_64.so
16/02/23 18:35:15 INFO GPRegression: Generating error bars
16/02/23 18:35:15 INFO RegressionMetrics: Regression Model Performance
16/02/23 18:35:15 INFO RegressionMetrics: ============================
16/02/23 18:35:15 INFO RegressionMetrics: MAE: 0.832018817808599
16/02/23 18:35:15 INFO RegressionMetrics: RMSE: 1.2904097720941374
16/02/23 18:35:15 INFO RegressionMetrics: RMSLE: 0.10885967880476728
16/02/23 18:35:15 INFO RegressionMetrics: R^2: 0.9339831074509592
16/02/23 18:35:15 INFO RegressionMetrics: Corr. Coefficient: 0.9731513401331606
16/02/23 18:35:15 INFO RegressionMetrics: Model Yield: 0.8073520083122128
16/02/23 18:35:15 INFO RegressionMetrics: Std Dev of Residuals: 1.270213452763595
16/02/23 18:35:15 INFO RegressionMetrics: Generating Plot of Residuals
16/02/23 18:35:15 INFO RegressionMetrics: Generating plot of residuals vs labels


```

In this example `TestGPDelve` we train a GP model based on the RBF Kernel with its bandwidth/length scale set to `2.0` and the noise level set to `1.0`, we use 500 input output patterns to train and test on an independent sample of 1000 data points. Apart from printing a bunch of evaluation metrics in the console DynaML also generates Javascript plots using Wisp in the browser.

![plots1](https://cloud.githubusercontent.com/assets/1389553/13259040/ff9bfa84-da55-11e5-9325-f58a73ebf532.png)
