package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 15/12/15.
  */
object TestGPHousing {

  def apply(kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            bandwidth: Double = 0.5,
            noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            trainFraction: Double = 0.75,
            columns: List[Int] = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12),
            grid: Int = 5, step: Double = 0.2, globalOpt: String = "ML",
            stepSize: Double = 0.01, maxIt: Int = 300): Unit =
    runExperiment(kernel, bandwidth,
      noise, (506*trainFraction).toInt, columns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString
      )
    )

  def apply(kern: String,
            bandwidth: Double,
            noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            trainFraction: Double,
            columns: List[Int],
            grid: Int, step: Double, globalOpt: String,
            stepSize: Double, maxIt: Int): Unit = {

    val kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
      kern match {
        case "RBF" =>
          new RBFKernel(bandwidth)
        case "Cauchy" =>
          new CauchyKernel(bandwidth)
        case "Laplacian" =>
          new LaplacianKernel(bandwidth)
        case "RationalQuadratic" =>
          new RationalQuadraticKernel(bandwidth)
        case "FBM" => new FBMKernel(bandwidth)
        case "Student" => new TStudentKernel(bandwidth)
        case "Periodic" => new PeriodicKernel(bandwidth, bandwidth)
      }

    val num_training = 506*trainFraction

    runExperiment(kernel, bandwidth,
      noise, num_training.toInt, columns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString
      )
    )

  }

  def runExperiment(kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                    bandwidth: Double = 0.5,
                    noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                    num_training: Int = 200, columns: List[Int] = List(40,16,21,23,24,22,25),
                    grid: Int = 5, step: Double = 0.2,
                    globalOpt: String = "ML", opt: Map[String, String]): Unit = {

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

        metrics.setName("MEDV")

        //println(scoresAndLabels)
        metrics.print()
        metrics.generatePlots()
      }

    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, 506-num_training) >
      DynaMLPipe.gaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/housing.data", "data/housing.data"))

  }

}
