package io.github.tailhq.dynaml.examples

import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.analysis.VectorField
import io.github.tailhq.dynaml.evaluation.RegressionMetrics
import io.github.tailhq.dynaml.kernels._
import io.github.tailhq.dynaml.pipes._
import io.github.tailhq.dynaml.DynaMLPipe._
import io.github.tailhq.dynaml.modelpipe.GPRegressionPipe
import io.github.tailhq.dynaml.utils.GaussianScaler
import io.github.tailhq.dynaml.examples._

/**
  * Example: Gaussian Process regression on fried delve data.
  */
object TestGPDelve {

  type Features = DenseVector[Double]
  type Data = Iterable[(Features, Features)]
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

    val preScaling = IterableDataPipe(
      (pattern: (Features, Double)) => (pattern._1, DenseVector(pattern._2))
    )

    val postScaling = IterableDataPipe(
      (pattern: (Features, Features)) => (pattern._1, pattern._2(0))
    )


    val modelPipe = GPRegressionPipe(
      (d: Data) => d.map(p => (p._1, p._2(0))).toSeq,
      kernel, new DiracKernel(noise))


    val modelTrainTest =
      (trainTest: DataAndScales) => {
        val (training, test, scales): DataAndScales = trainTest

        val model = modelPipe(training)
        val res = model.test(postScaling.run(test).toSeq).map(t => (DenseVector(t._3), DenseVector(t._2)))

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

    val dataFile = dataDir+"/delve.csv"
    trainTestPipe.run((dataFile, dataFile))

  }

}
