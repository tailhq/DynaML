---
layout: page
title: DynaML Shell Examples
---

## DynaML REPL


The DynaML scala shell is the first point of contact when experimenting with data analysis workflows and modeling algorithms. In this section we consider two representative examples of binary classification and regression.

## Model Building

After starting the DynaML REPL one can start experimenting with building models for data sets. In the `data/` directory you can find a set of packaged data sets. We will work with the file `housing.csv` for now.


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
	Create a list of column numbers to
	extract from the data file
	the first one being the target variable. 
	Next create two preprocessing work flows
	1. Replace all blank spaces with commas
	2. Using the library function utils.extractColumns
	   to get the required columns.
*/

val columns = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12)
val num_training = 506*trainFraction
val replaceWhiteSpaces = (s: Stream[String]) =>
	s.map(utils.replace("\\s+")(","))
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


    val normalizationFunc =
	(point: (DenseVector[Double], Double)) => {
      val extendedpoint =
		  DenseVector(point._1.toArray ++
		  Array(point._2))

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
	  new GPRegression(kernel, trainTest._1._1.toSeq)
		  .setNoiseLevel(noise)

	val res = model.test(trainTest._1._2.toSeq)


	val scoresAndLabelsPipe =
      DataPipe((res: Seq[(DenseVector[Double], Double,
	    Double, Double, Double)]) =>
          res.map(i => (i._3, i._2)).toList) >
	  DataPipe((list: List[(Double, Double)]) =>
        list.map{l =>
		(l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
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
