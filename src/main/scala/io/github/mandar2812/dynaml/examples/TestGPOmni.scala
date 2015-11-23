package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 19/11/15.
  */
object TestGPOmni {
  def apply(year: Int = 2006, kern: String = "RBF",
            bandwidth: Double = 0.5, noise: Double = 0.0,
            num_training: Int = 200, num_test: Int = 50,
            columns: List[Int] = List(40,16,21,23,24,22,25)): Unit = {

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
      }
    //val vectorizeRecordPipe = StreamDataPipe((tup: (DenseVector[Double], Double)) =>
    //DenseVector(tup._1.toArray ++ Array(tup._2)))
    //val identityPipe = DataPipe(identity[(DenseVector[Double], Double)])
    //val parallelPipe = DataPipe(vectorizeRecordPipe, identityPipe)
    //val normalizePipe = DataPipe((l: Stream[DenseVector[Double]]) => utils.getStats(l.toList))

    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val replaceWhiteSpaces = (s: Stream[String]) => s.map(utils.replace("\\s+")(","))

    val filterMissingValues = (lines: Stream[String]) => lines.filter(line => !line.contains(",,"))

    val extractDstTimeSeries = (lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = splits(1).toDouble * 24 + splits(2).toDouble
      (timestamp, splits(3).toDouble)
    }

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", columns,
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9"))

    val splitTrainingTest = (data: Stream[(DenseVector[Double], Double)]) => {
      (data.take(num_training), data.take(num_training+num_test).takeRight(num_test))
    }

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
        val model = new GPRegression(kernel, trainTest._1._1.toSeq).setNoiseLevel(noise)
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

    val processpipe = DataPipe(utils.textFileToStream _) >
      DataPipe(replaceWhiteSpaces) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => !line.contains(",,")) >
      StreamDataPipe((line: String) => {
        val split = line.split(",")
        (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
      }) >
      /*DataPipe(extractDstTimeSeries) >*/
      DataPipe(splitTrainingTest) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    processpipe.run("data/omni2_"+year+".csv")

  }

}
