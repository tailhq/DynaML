package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * Example: Gaussian Process regression on fried delve data.
  */
object TestGPDelve {
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

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", columns, Map())

    val normalizeData =
      (trainTest: (Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)])) => {

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

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPRegression(kernel, new DiracKernel(noise), trainTest._1._1.toSeq)
        val res = model.test(trainTest._1._2)
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
        //metrics.generatePlots()
      }

    val preProcessPipe = DataPipe(utils.textFileToStream _) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => {
        val split = line.split(",")
        (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
      })

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DataPipe((data: (Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)])) => {
        (data._1.take(training),
          data._2.takeRight(test))
      }) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/delve.csv", "data/delve.csv"))

  }

}
