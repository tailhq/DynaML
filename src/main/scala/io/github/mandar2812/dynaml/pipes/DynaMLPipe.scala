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

  /**
    * Data pipe which takes a file name/path as a
    * [[String]] and returns a [[Stream]] of [[String]].
    * */
  val fileToStream = DataPipe(utils.textFileToStream _)

  /**
    * Writes a [[Stream]] of [[String]] to
    * a file.
    *
    * Usage: DynaMLPipe.streamToFile("abc.csv")
    * */
  val streamToFile = (fileName: String) => DataPipe(utils.writeToFile(fileName) _)

  /**
    * Data pipe to replace all white spaces in a [[Stream]]
    * of [[String]] with the comma character.
    * */
  val replaceWhiteSpaces = StreamDataPipe((s: String) => utils.replace("\\s+")(",")(s))

  /**
    * Trim white spaces from each line in a [[Stream]]
    * of [[String]]
    * */
  val trimLines = StreamDataPipe((s: String) => s.trim())

  /**
    * This pipe assumes its input to be of the form
    * "YYYY,Day,Hour,Value"
    *
    * It takes as input a function (TFunc) which
    * converts a [[Tuple3]] into a single "timestamp" like value.
    *
    * The pipe processes its data source line by line
    * and outputs a [[Tuple2]] in the following format.
    *
    * (Timestamp,Value)
    *
    * Usage: DynaMLPipe.extractTimeSeries(TFunc)
    * */
  val extractTimeSeries = (Tfunc: (Double, Double, Double) => Double) =>
    DataPipe((lines: Stream[String]) => lines.map{line =>
    val splits = line.split(",")
    val timestamp = Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
    (timestamp, splits(3).toDouble)
  })

  /**
    * This pipe is exactly similar to [[DynaMLPipe.extractTimeSeries]],
    * with one key difference, it returns a [[Tuple2]] like
    * (Timestamp, FeatureVector), where FeatureVector is
    * a Vector of values.
    * */
  val extractTimeSeriesVec = (Tfunc: (Double, Double, Double) => Double) =>
    DataPipe((lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
      val feat = DenseVector(splits.slice(3, splits.length).map(_.toDouble))
      (timestamp, feat)
    })

  /**
    * Inorder to generate features for auto-regressive models,
    * one needs to construct sliding windows in time. This function
    * takes two parameters
    *
    * deltaT: the auto-regressive order
    * timelag: the time lag after which the windowing is conducted.
    *
    * E.g
    *
    * Let deltaT = 2 and timelag = 1
    *
    * This pipe will take stream data of the form
    * (t, Value_t)
    *
    * and output a stream which looks like
    *
    * (t, Vector(Value_t-2, Value_t-3))
    *
    * */
  val deltaOperation = (deltaT: Int, timelag: Int) =>
    DataPipe((lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(deltaT+timelag+1).map((history) => {
        val features = DenseVector(history.take(deltaT).map(_._2).toArray)
        (features, history.last._2)
      }).toStream)

  /**
    * The vector version of [[DynaMLPipe.deltaOperation]]
    * */
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

  /**
    * The vector ARX version of [[DynaMLPipe.deltaOperation]]
    * */
  val deltaOperationARX = (deltaT: List[Int]) =>
    DataPipe((lines: Stream[(Double, DenseVector[Double])]) =>
      lines.toList.sliding(deltaT.max+1).map((history) => {
        val hist = history.take(history.length - 1).map(_._2)
        val featuresAcc: ML[Double] = ML()

        (0 until hist.head.length).foreach((dimension) => {
          //for each dimension/regressor take points t to t-order
          featuresAcc ++= hist.takeRight(deltaT(dimension))
            .map(vec => vec(dimension))
        })

        val features = DenseVector(featuresAcc.toArray)
        //assert(history.length == deltaT + 1, "Check one")
        //assert(features.length == deltaT, "Check two")
        (features, history.last._2(0))
      }).toStream)

  /**
    * From a [[Stream]] of [[String]] remove all records
    * which contain missing values, this pipe should be applied
    * after the application of [[DynaMLPipe.extractTrainingFeatures]].
    * */
  val removeMissingLines = StreamDataPipe((line: String) => !line.contains("<NA>"))

  /**
    * Take each line which is a comma separated string and extract
    * all but the last element into a feature vector and leave the last
    * element as the "target" value.
    *
    * This pipe outputs data in a [[Stream]] of [[Tuple2]] in the following form
    *
    * (Vector(features), value)
    * */
  val splitFeaturesAndTargets = StreamDataPipe((line: String) => {
    val split = line.split(",")
    (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
  })

  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  val gaussianStandardization =
    DataPipe((trainTest: (Stream[(DenseVector[Double], Double)],
      Stream[(DenseVector[Double], Double)])) => {
      logger.info("Training Data")
      logger.info(trainTest._1.toList)
      logger.info("-----------")
      logger.info("Test Data")
      logger.info(trainTest._2.toList)
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


  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  val gaussianStandardizationMO =
    DataPipe((trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
      Stream[(DenseVector[Double], DenseVector[Double])])) => {
      //logger.info("Training Data")
      //logger.info(trainTest._1.toList)
      //logger.info("-----------")
      //logger.info("Test Data")
      //logger.info(trainTest._2.toList)
      val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
        DenseVector(tup._1.toArray ++ tup._2.toArray)).toList)

      val stdDev: DenseVector[Double] = variance.map(v =>
        math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


      val normalizationFunc = (point: (DenseVector[Double], DenseVector[Double])) => {
        val extendedpoint = DenseVector(point._1.toArray ++ point._2.toArray)

        val normPoint = (extendedpoint - mean) :/ stdDev
        val length = point._1.length
        val outlength = point._2.length
        //logger.info("Features Normalized: "+normPoint(0 until length))
        //logger.info("Outputs Normalized: "+normPoint(length until length+outlength))
        (normPoint(0 until length), normPoint(length until length+outlength))
      }

      ((trainTest._1.map(normalizationFunc),
        trainTest._2.map(normalizationFunc)), (mean, stdDev))
    })

  /**
    * Extract a subset of the data into a [[Tuple2]] which
    * can be used as a training, test combo for model learning and evaluation.
    *
    * Usage: DynaMLPipe.splitTrainingTest(num_training, num_test)
    * */
  val splitTrainingTest = (num_training: Int, num_test: Int) =>
    DataPipe((data: (Stream[(DenseVector[Double], Double)],
    Stream[(DenseVector[Double], Double)])) => {
    (data._1.take(num_training), data._2.takeRight(num_test))
  })

  /**
    * Extract a subset of columns from a [[Stream]] of comma separated [[String]]
    * also replace any missing value strings with the empty string.
    *
    * Usage: DynaMLPipe.extractTrainingFeatures(List(1,2,3), Map(1 -> "N.A.", 2 -> "NA", 3 -> "na"))
    * */
  val extractTrainingFeatures =
    (columns: List[Int], m: Map[Int, String]) => DataPipe((l: Stream[String]) =>
    utils.extractColumns(l, ",", columns, m))

  /**
    * Takes a base pipe and creates a parallel pipe by duplicating it.
    *
    * @param pipe The base data pipe
    * @return a [[ParallelPipe]] object.
    * */
  def duplicate[Source, Destination](pipe: DataPipe[Source, Destination]) =
    DataPipe(pipe, pipe)

  def GPTune[M <: AbstractGPRegressionModel[
    Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](globalOpt: String,
                          grid: Int, step: Double) =
    DataPipe((model: M) => {
      val gs = globalOpt match {
        case "GS" => new GridSearch[M](model)
          .setGridSize(grid)
          .setStepSize(step)
          .setLogScale(false)

        case "ML" => new GPMLOptimizer[DenseVector[Double],
          Seq[(DenseVector[Double], Double)], M](model)
      }

      val startConf = model.covariance.state ++ model.noiseModel.state
      gs.optimize(startConf, Map("tolerance" -> "0.0001",
        "step" -> step.toString,
        "maxIterations" -> grid.toString))
    })

  def GPRegressionTest[T <: AbstractGPRegressionModel[
    Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](model:T) =
    DataPipe(
    (trainTest: (Stream[(DenseVector[Double], Double)],
      (DenseVector[Double], DenseVector[Double]))) => {

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
