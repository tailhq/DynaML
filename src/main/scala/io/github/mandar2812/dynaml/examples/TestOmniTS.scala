package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPTimeSeries, GPRegression}
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 22/11/15.
  */
object TestOmniTS {
  def apply(year: Int = 2006, kern: String = "RBF",
            bandwidth: Double = 0.5, noise: Double = 0.0,
            num_training: Int = 200, num_test: Int = 50,
            column: Int = 40): Unit = {

    val kernel: CovarianceFunction[Double, Double, DenseMatrix[Double]] =
      kern match {
        case "RBF" =>
          new RBFCovFunc(bandwidth)
        case "Cauchy" =>
          new CauchyCovFunc(bandwidth)
        case "Laplacian" =>
          new LaplaceCovFunc(bandwidth)
        case "RationalQuadratic" =>
          new RationalQuadraticCovFunc(bandwidth)
        case "FBM" =>
          new FBMCovFunction(bandwidth)
        case "Wave" =>
          new WaveCovFunc(bandwidth)
        case "Identity" =>
          new IdentityCovFunc
        case "Student" =>
          new TStudentCovFunc(bandwidth)
        case "Wavelet" =>
          new WaveletCovFunc((x) => math.cos(1.75*x)*math.exp(-1.0*x*x/2.0))(bandwidth)
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

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", List(0,1,2,column),
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9"))

    val extractTimeSeries = (lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = splits(1).toDouble * 24 + splits(2).toDouble
      (timestamp, splits(3).toDouble)
    }

    val splitTrainingTest = (data: Stream[(Double, Double)]) => {
      (data.take(num_training), data.take(num_training+num_test).takeRight(num_test))
    }

    val normalizeData =
      (trainTest: (Stream[(Double, Double)], Stream[(Double, Double)])) => {

        val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
          DenseVector(tup._1, tup._2)).toList)

        val stdDev: DenseVector[Double] = variance.map(v =>
          math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


        val normalizationFunc = (point: (Double, Double)) => {

          val normPoint = (DenseVector(point._1, point._2) - mean) :/ stdDev
          (point._1, normPoint(1))
        }

        ((trainTest._1.map(normalizationFunc),
          trainTest._2.map(normalizationFunc)), (mean, stdDev))
      }

    val modelTrainTest =
      (trainTest: ((Stream[(Double, Double)],
        Stream[(Double, Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPTimeSeries(kernel, trainTest._1._1.toSeq).setNoiseLevel(noise)
        val res = model.test(trainTest._1._2.toSeq)
        val scoresAndLabelsPipe =
          DataPipe((res: Seq[(Double, Double, Double, Double, Double)]) =>
            res.map(i => (i._3, i._2)).toList) >
            DataPipe((list: List[(Double, Double)]) =>
              list.map{l => (l._1*trainTest._2._2(1) + trainTest._2._1(1),
                l._2*trainTest._2._2(1) + trainTest._2._1(1))}
            )

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
      DataPipe(extractTimeSeries) >
      DataPipe(splitTrainingTest) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    processpipe.run("data/omni2_"+year+".csv")

  }
}
