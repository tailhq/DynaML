package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.modelpipe.GPRegressionPipe
import io.github.mandar2812.dynaml.utils.GaussianScaler

/**
  * Example: Gaussian Process regression on fried delve data.
  */
object TestGPDelve {

  type Features = DenseVector[Double]
  type Data = Stream[(Features, Features)]
  type Scales = (GaussianScaler, GaussianScaler)
  type DataAndScales = (Data, Data, Scales)

  def apply (kern: String = "RBF",
             bandwidth: Double = 0.5, noise: Double = 0.0,
             training: Int = 100, test: Int = 1000,
             columns: List[Int] = List(10,0,1,2,3,4,5,6,7,8,9)): Unit = {

    implicit val field = VectorField(columns.length - 1)

    val kernel: LocalScalarKernel[DenseVector[Double]] =
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
      }

    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preScaling = StreamDataPipe(
      (pattern: (Features, Double)) => (pattern._1, DenseVector(pattern._2))
    )

    val postScaling = StreamDataPipe(
      (pattern: (Features, Features)) => (pattern._1, pattern._2(0))
    )


    val modelPipe = GPRegressionPipe(
      (d: Data) => d.map(p => (p._1, p._2(0))).toSeq,
      kernel, new DiracKernel(noise))


    val modelTrainTest =
      (trainTest: DataAndScales) => {
        val (training, test, scales) = trainTest

        val model = modelPipe(training)
        val res = model.test(postScaling(test)).map(t => (DenseVector(t._3), DenseVector(t._2)))

        val scoresAndLabels = (scales._2.i * scales._2.i)(res)

        val metrics = new RegressionMetrics(
          scoresAndLabels.map(p => (p._1(0), p._2(0))).toList,
          scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()

        (model, metrics)
      }

    val preProcessPipe = fileToStream >
      extractTrainingFeatures(columns, Map()) >
      splitFeaturesAndTargets >
      preScaling


    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      splitTrainingTest(training, test) >
      gaussianScalingTrainTest >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/delve.csv", "data/delve.csv"))

  }

}
