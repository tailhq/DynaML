---
layout: default
title: REPL Examples
noToc: true
---
## DynaML REPL


The DynaML scala shell is the first point of contact when experimenting with data analysis workflows and modeling algorithms. In this section we consider two representative examples of binary classification and regression.

## Model Building

After starting the DynaML REPL one can start experimenting with building models for data sets. In the `data/` directory you can find a set of packaged data sets. We will work with the files `ripley.csv` and `housing.csv` for now.


### Regression: Boston Housing Data Set

The [boston housing](https://archive.ics.uci.edu/ml/datasets/Housing) data set is a popular multivariate regression set encountered in machine learning research. In this section we train demonstrate how to train a *Gaussian Process* regression model on this data. For a detailed introduction on *Gaussian Processes* you can refer to the book written by [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en).

In this example we make use of the *pipes* module of DynaML which enables us to plug together arbitrary number of data processing operations. This allows us to separate data pre-processing actions from the actual model building.

We start by
```scala

	import io.github.mandar2812.dynaml.models.gp.GPRegression
	import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
	val trainFraction = 0.5
	val noise = 1.0

	/*
		Create a list of column numbers to extract from the data file
		the first one being the target variable. 
		Next create two preprocessing work flows
		1. Replace all blank spaces with commas
		2. Using the library function utils.extractColumns
		   to get the required columns.
	*/
	val columns = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12)
	val num_training = 506*trainFraction

    val replaceWhiteSpaces = (s: Stream[String]) => s.map(utils.replace("\\s+")(","))

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", columns, Map())
```


```scala
	val normalizeData =
      (trainTest: (Stream[(DenseVector[Double], Double)],
	  Stream[(DenseVector[Double], Double)])) => {

        val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
          DenseVector(tup._1.toArray ++ Array(tup._2))).toList)

        val stdDev: DenseVector[Double] = variance.map(v =>
          math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


        val normalizationFunc = (point: (DenseVector[Double], Double)) => {
          val extendedpoint = DenseVector(point._1.toArray ++ Array(point._2))

          val normPoint = (extendedpoint - mean) :/ stdDev
          val length = normPoint.length
          (normPoint(0 until length), normPoint(-1))
        }

        ((trainTest._1.map(normalizationFunc),
          trainTest._2.map(normalizationFunc)), (mean, stdDev))
      }
```

```scala
	val kernel = new RBFKernel(2.5)
	
	val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

	    val model =
		new GPRegression(kernel, trainTest._1._1.toSeq).setNoiseLevel(noise)

	    val res = model.test(trainTest._1._2.toSeq)


	    val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double,
			Double, Double, Double)]) =>
              res.map(i => (i._3, i._2)).toList) >
			  DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

	    val scoresAndLabels = scoresAndLabelsPipe.run(res)

	    val metrics = new RegressionMetrics(scoresAndLabels,
		scoresAndLabels.length)

        
		metrics.print()
		metrics.generatePlots()
	}
```

```scala
	
	val preProcessPipe = DataPipe(utils.textFileToStream _) >
      DataPipe((s: Stream[String]) => s.map(_.trim())) >
      DataPipe(replaceWhiteSpaces) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => {
        val split = line.split(",")
        (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
      })

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DataPipe((data: (Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)])) => {
        (data._1.take(num_training.toInt),
		 data._2.takeRight(506-num_training.toInt))
      }) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/housing.data", "data/housing.data"))
```

***


### Binary Classification: Ripley Data Set

Now we train a *Least Squares Support Vector Machine* (LSSVM) model for binary classification on the ripley data set. The LSSVM is a least squares formulation of the classical SVM algorithm. For a detailed treatment on LSSVMs, you may refer to the seminal work on the subject by [Suykens and De Moor](https://books.google.nl/books?id=g8wEimyEmrUC&printsec=frontcover&dq=least+squares+support+vector&hl=en&sa=X&ved=0ahUKEwipouOk67HKAhVBIg8KHf2JB2YQ6AEIIDAA).

We start by specifying some configuration variables required for the creation of the LSSVM model.

```scala
	val config = Map("file" -> "data/ripley.csv",
	"delim" -> ",", "head" -> "false",
	"task" -> "classification")
	val model = LSSVMModel(config)
```

We can now (optionally) add a kernel on the model to create a kernel based Least Squares Support Vector Machine (LSSVM) model.

```scala
  val rbf = new RBFKernel(1.025)
  model.applyKernel(rbf)
```

```
15/08/03 19:07:42 INFO GreedyEntropySelector$: Returning final prototype set
15/08/03 19:07:42 INFO SVMKernel$: Constructing key-value representation of kernel matrix.
15/08/03 19:07:42 INFO SVMKernel$: Dimension: 13 x 13
15/08/03 19:07:42 INFO SVMKernelMatrix: Eigenvalue decomposition of the kernel matrix using JBlas.
15/08/03 19:07:42 INFO SVMKernelMatrix: Eigenvalue stats: 0.09104374173019622 =< lambda =< 3.110068839504519
15/08/03 19:07:42 INFO LSSVMModel: Applying Feature map to data set
15/08/03 19:07:42 INFO LSSVMModel: DONE: Applying Feature map to data set
DynaML>
```

Now we can solve the optimization problem posed by the LS-SVM in the parameter space. Since the LS-SVM problem is equivalent to ridge regression, we have to specify a regularization constant.

```scala
  model.setRegParam(1.5).learn
```

We can now predict the value of the target variable given a new point consisting of a Vector of features using `model.predict()`. Evaluating models is easy in DynaML. You can create an evaluation object as follows. 

```scala
	val configtest = Map("file" -> "data/ripleytest.csv",
	"delim" -> ",", "head" -> "false")
	val met = model.evaluate(configtest)
	met.print
```
The object `met` has a `print()` method which will dump some performance metrics in the shell. But you can also generate plots by using the `generatePlots()` method.

```
15/08/03 19:08:40 INFO BinaryClassificationMetrics: Classification Model Performance
15/08/03 19:08:40 INFO BinaryClassificationMetrics: ============================
15/08/03 19:08:40 INFO BinaryClassificationMetrics: Accuracy: 0.6172839506172839
15/08/03 19:08:40 INFO BinaryClassificationMetrics: Area under ROC: 0.2019607843137254
```

```scala
met.generatePlots
```
