package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPNarxModel, GPRegression}
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils
import com.quantifind.charts.Highcharts._
import org.apache.log4j.Logger

/**
  * Created by mandar on 22/11/15.
  */
object TestOmniAR {

  def apply(year: Int, kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            delta: Int, bandwidth: Double, noise: Double,
            num_training: Int, num_test: Int,
            column: Int, grid: Int,
            step: Double, globalOpt: String,
            stepSize: Double,
            maxIt: Int): Unit =
    runExperiment(year, kernel, delta, bandwidth, noise,
      num_training, num_test, column, grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString))

  def apply(year: Int = 2006, kern: String = "RBF",
            delta: Int = 2, bandwidth: Double = 0.5, noise: Double = 0.0,
            num_training: Int = 200, num_test: Int = 50,
            column: Int = 40, grid: Int = 5,
            step: Double = 0.2, globalOpt: String,
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
        case "Anova" => new AnovaKernel(bandwidth)
      }
    runExperiment(year, kernel, delta, bandwidth, noise, num_training,
      num_test, column, grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString))
  }

  def runExperiment(year: Int = 2006,
                    kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                    deltaT: Int = 2, bandwidth: Double = 0.5, noise: Double = 0.0,
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, grid: Int = 5,
                    step: Double = 0.2, globalOpt: String = "ML",
                    opt: Map[String, String]): Unit = {
    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val logger = Logger.getLogger(this.getClass)
    val replaceWhiteSpaces = (s: Stream[String]) => s.map(utils.replace("\\s+")(","))

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", List(0,1,2,column),
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999"))

    val extractTimeSeries = (lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = splits(1).toDouble * 24 + splits(2).toDouble
      (timestamp, splits(3).toDouble)
    }


    val deltaOperation = (lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(deltaT+1).map((history) => {
        val features = DenseVector(history.take(history.length - 1).map(_._2).toArray)
        (features, history.last._2)
    }).toStream

    val splitTrainingTest = (data: Stream[(DenseVector[Double], Double)]) => {
      (data.take(num_training), data.take(num_training+num_test).takeRight(num_test))
    }

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
          (normPoint(0 until length-1), normPoint(-1))
        }

        ((trainTest._1.map(normalizationFunc),
          trainTest._2.map(normalizationFunc)), (mean, stdDev))
      }

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPNarxModel(deltaT, kernel, trainTest._1._1.toSeq).setNoiseLevel(noise)

        val gs = globalOpt match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)

          case "ML" => new GPMLOptimizer[DenseVector[Double],
            Seq[(DenseVector[Double], Double)],
            GPRegression](model)
        }

        val startConf = kernel.state ++ Map("noiseLevel" -> noise)
        val (_, conf) = gs.optimize(kernel.state + ("noiseLevel" -> noise), opt)

        model.setState(conf)

        val res = model.test(trainTest._1._2.toSeq)

        val deNormalize = DataPipe((list: List[(Double, Double)]) =>
          list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2)).toList) > deNormalize

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)


        logger.info("Printing One Step Ahead (OSA) Performance Metrics")
        metrics.print()
        metrics.generatePlots()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        legend(List("Time Series", "Predicted Time Series (one hour ahead)"))
        unhold()


        /*logger.info("Printing Model Predicted Output (MPO) Performance Metrics")
        //Now test the Model Predicted Output and its performance.
        val mpo = model.modelPredictedOutput(trainTest._1._2.length) _
        val predictedOutput = mpo(trainTest._1._2.head._1)
          .map(_._1)
        val outputs = trainTest._1._2.map(_._2).toList

        val res2 = predictedOutput zip outputs

        val mpoMetrics = new RegressionMetrics(deNormalize.run(res2.toList),
          res2.length)
        mpoMetrics.print()
        mpoMetrics.generatePlots()*/

      }

    val processpipe = DataPipe(utils.textFileToStream _) >
      DataPipe(replaceWhiteSpaces) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => !line.contains(",,")) >
      DataPipe(extractTimeSeries) >
      DataPipe(deltaOperation) >
      DataPipe(splitTrainingTest) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    processpipe.run("data/omni2_"+year+".csv")


  }
}