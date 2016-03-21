package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils

import scala.util.Random

/**
  * @author mandar2812 datum 19/11/15.
  *
  * Train and evaluate a "vanilla"
  * GP regression model f(x): R_n --> R
  */
object TestGPOmni {

  def apply (kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
             year: Int, yeartest: Int,
             bandwidth: Double,
             noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
             num_training: Int,
             num_test: Int, columns: List[Int],
             grid: Int, step: Double,
             randomSample: Boolean,
             globalOpt: String,
             stepSize: Double,
             maxIt: Int): Unit = {

    runExperiment(year, yeartest, kernel, bandwidth,
      noise, num_training, num_test, columns,
      grid, step, globalOpt, randomSample,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString))

  }

  def runExperiment(year: Int = 2006, yeartest: Int = 2007,
                    kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                    bandwidth: Double = 0.5,
                    noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                    num_training: Int = 200, num_test: Int = 50,
                    columns: List[Int] = List(40,16,21,23,24,22,25),
                    grid: Int = 5, step: Double = 0.2,
                    globalOpt: String = "ML", randomSample: Boolean = false,
                    opt: Map[String, String]): Unit = {



    //function to train and test a GP Regression model
    //accepts training and test splits separately.
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPRegression(kernel, noise, trainingdata = trainTest._1._1.toSeq)

        val gs = globalOpt match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)

          case "ML" => new GPMLOptimizer[DenseVector[Double],
            Seq[(DenseVector[Double], Double)],
            GPRegression](model)
        }

        val startConf = kernel.state ++ noise.state
        val (_, conf) = gs.optimize(startConf, opt)

        model.setState(conf)

        val res = model.test(trainTest._1._2.toSeq)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2)).toList) > DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        //println(scoresAndLabels)
        metrics.print()
        metrics.generatePlots()

      }

    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(columns,
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999")) >
      DynaMLPipe.removeMissingLines >
      DynaMLPipe.splitFeaturesAndTargets

    /*
    * Create the final pipe composed as follows
    *
    * train, test
    *   |       |
    *   |-------|
    *   |       |
    *   v       v
    * p_train, p_test : pre-process
    *   |       |
    *   |-------|
    *   |       |
    *   v       v
    * s_train, s_test : sub-sample
    *   |       |
    *   |-------|
    *   |       |
    *   v       v
    * norm_tr, norm_te : mean center and standardize
    *   |       |
    *   |-------|
    *   |       |
    *   v       v
    *   |       |
    *  |-----------------|
    *  | Train, tune and |
    *  | test the model. |
    *  | Output graphs,  |
    *  | metrics         |
    *  |_________________|
    *
    * */
    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DataPipe((data: (Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)])) => {
        if(!randomSample)
          (data._1.take(num_training), data._2.takeRight(num_test))
        else
          (data._1.filter(_ => Random.nextDouble() <= num_training/data._1.size.toDouble),
            data._2.filter(_ => Random.nextDouble() <= num_test/data._2.size.toDouble))
      }) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)


    trainTestPipe.run(("data/omni2_"+year+".csv", "data/omni2_"+yeartest+".csv"))
  }

}
