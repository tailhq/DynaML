package io.github.mandar2812.dynaml.pipes

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

import scala.collection.mutable.{MutableList => ML}

/**
  * @author mandar2812 datum 3/2/16.
  *
  * A library of sorts for common data processing
  * pipes.
  */
object DynaMLPipe {


  val logger = Logger.getLogger(this.getClass)

  val fileToStream = DataPipe(utils.textFileToStream _)

  val replaceWhiteSpaces = DataPipe((s: Stream[String]) => s.map(utils.replace("\\s+")(",")))

  val trimLines = DataPipe((s: Stream[String]) => s.map(_.trim()))

  val deltaOperation = (deltaT: Int, timelag: Int) =>
    DataPipe((lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(deltaT+timelag+1).map((history) => {
        val features = DenseVector(history.take(deltaT).map(_._2).toArray)
        (features, history.last._2)
      }).toStream)

  val deltaOperationVec = (deltaT: Int) =>
    DataPipe((lines: Stream[(Double, DenseVector[Double])]) =>
    lines.toList.sliding(deltaT+1).map((history) => {
      val hist = history.take(history.length - 1).map(_._2)
      val featuresAcc: ML[Double] = ML()

      (0 until hist.head.length).foreach((dimension) => {
        //for each dimension/regressor take points t to t-order
        featuresAcc ++= hist.map(vec => vec(dimension))
      })

      val features = DenseVector(featuresAcc.toArray)
      //assert(history.length == deltaT + 1, "Check one")
      //assert(features.length == deltaT, "Check two")
      (features, history.last._2(0))
    }).toStream)

  val extractTimeSeries = (Tfunc: (Double, Double, Double) => Double) =>
    DataPipe((lines: Stream[String]) => lines.map{line =>
    val splits = line.split(",")
    val timestamp = Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
    (timestamp, splits(3).toDouble)
  })


  val extractTimeSeriesVec = (Tfunc: (Double, Double, Double) => Double) =>
    DataPipe((lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
      val feat = DenseVector(splits.slice(3, splits.length).map(_.toDouble))
      (timestamp, feat)
    })


  val removeMissingLines = StreamDataPipe((line: String) => !line.contains(",,"))

  val splitFeaturesAndTargets = StreamDataPipe((line: String) => {
    val split = line.split(",")
    (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
  })

  val gaussianStandardization =
    DataPipe((trainTest: (Stream[(DenseVector[Double], Double)],
      Stream[(DenseVector[Double], Double)])) => {

      logger.info(trainTest._1.toList)

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
    })

  val splitTrainingTest = (num_training: Int, num_test: Int) =>
    DataPipe((data: (Stream[(DenseVector[Double], Double)],
    Stream[(DenseVector[Double], Double)])) => {
    (data._1.take(num_training), data._2.takeRight(num_test))
  })

  val extractTrainingFeatures =
    (columns: List[Int], m: Map[Int, String]) => DataPipe((l: Stream[String]) =>
    utils.extractColumns(l, ",", columns, m))

  def duplicate[Source, Destination](pipe: DataPipe[Source, Destination]) =
    DataPipe(pipe, pipe)

  def GPRegressionTest[T <: AbstractGPRegressionModel[
    Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](model:T, globalOpt: String,
                          grid: Int, step: Double) =
    DataPipe(
    (trainTest: (Stream[(DenseVector[Double], Double)],
      (DenseVector[Double], DenseVector[Double]))) => {

      val gs = globalOpt match {
        case "GS" => new GridSearch[model.type](model)
          .setGridSize(grid)
          .setStepSize(step)
          .setLogScale(false)

        case "ML" => new GPMLOptimizer[DenseVector[Double],
          Seq[(DenseVector[Double], Double)],
          model.type](model)
      }

      val startConf = model.covariance.state ++ model.noiseModel.state
      val (_, conf) = gs.optimize(startConf,
        Map("tolerance" -> "0.0001",
        "step" -> step.toString,
        "maxIterations" -> grid.toString))

      model.setState(conf)

      val res = model.test(trainTest._1.toSeq)
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
    })
}
