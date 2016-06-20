package io.github.mandar2812.dynaml

import scala.collection.mutable.{MutableList => ML}
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.ParameterizedLearner
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GPMLOptimizer, GloballyOptWithGrad, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DataPipe, ReversibleScaler, Scaler, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.{GaussianScaler, MinMaxScaler}
import org.apache.log4j.Logger

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
    * Writes a [[Stream]] of [[AnyVal]] to
    * a file.
    *
    * Usage: DynaMLPipe.valuesToFile("abc.csv")
    * */
  val valuesToFile = (fileName: String) => DataPipe((stream: Stream[Seq[AnyVal]]) =>
    utils.writeToFile(fileName)(stream.map(s => s.mkString(","))))

  /**
    * Drop the first element of a [[Stream]] of [[String]]
    * */
  val dropHead = DataPipe((s: Stream[String]) => s.tail)


  /**
    * Data pipe to replace all occurrences of a regular expression or string in a [[Stream]]
    * of [[String]] with with a specified replacement string.
    * */
  val replace = (original: String, newString: String) =>
    StreamDataPipe((s: String) => utils.replace(original)(newString)(s))

  /**
    * Data pipe to replace all white spaces in a [[Stream]]
    * of [[String]] with the comma character.
    * */
  val replaceWhiteSpaces = replace("\\s+", ",")

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
  @deprecated("*Standardization pipes are deprecated as of v1.4,"+
    " use pipes that output scaler objects instead")
  val trainTestGaussianStandardization =
    DataPipe((trainTest: (Stream[(DenseVector[Double], Double)],
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
    })


  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  @deprecated("*Standardization pipes are deprecated as of v1.4,"+
    " use pipes that output scaler objects instead")
  val featuresGaussianStandardization =
    DataPipe((trainTest: (Stream[(DenseVector[Double], Double)],
      Stream[(DenseVector[Double], Double)])) => {

      val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
        tup._1).toList)

      val stdDev: DenseVector[Double] = variance.map(v =>
        math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


      val normalizationFunc = (point: (DenseVector[Double], Double)) => {
        val normPoint = (point._1 - mean) :/ stdDev
        (normPoint, point._2)
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
  @deprecated("*Standardization pipes are deprecated as of v1.4,"+
    " use pipes that output scaler objects instead")
  val trainTestGaussianStandardizationMO =
    DataPipe((trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
      Stream[(DenseVector[Double], DenseVector[Double])])) => {

      val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
        DenseVector(tup._1.toArray ++ tup._2.toArray)).toList)

      val stdDev: DenseVector[Double] = variance.map(v =>
        math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


      val normalizationFunc = (point: (DenseVector[Double], DenseVector[Double])) => {
        val extendedpoint = DenseVector(point._1.toArray ++ point._2.toArray)

        val normPoint = (extendedpoint - mean) :/ stdDev
        val length = point._1.length
        val outlength = point._2.length

        (normPoint(0 until length),
          normPoint(length until length+outlength))
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
  val gaussianScalingTrainTest =
    DataPipe((trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
      Stream[(DenseVector[Double], DenseVector[Double])])) => {

      val (num_features, num_targets) = (trainTest._1.head._1.length, trainTest._1.head._2.length)

      val (mean, variance) = utils.getStats(trainTest._1.map(tup =>
        DenseVector(tup._1.toArray ++ tup._2.toArray)).toList)

      val stdDev: DenseVector[Double] = variance.map(v =>
        math.sqrt(v/(trainTest._1.length.toDouble - 1.0)))


      val featuresScaler = new GaussianScaler(mean(0 until num_features), stdDev(0 until num_features))

      val targetsScaler = new GaussianScaler(
        mean(num_features until num_features + num_targets),
        stdDev(num_features until num_features + num_targets))

      val scaler: ReversibleScaler[(DenseVector[Double], DenseVector[Double])] = featuresScaler * targetsScaler

      (scaler(trainTest._1), scaler(trainTest._2), (featuresScaler, targetsScaler))
    })

  /**
    * Perform [0,1] scaling on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  val minMaxScalingTrainTest =
    DataPipe((trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
      Stream[(DenseVector[Double], DenseVector[Double])])) => {

      val (num_features, num_targets) = (trainTest._1.head._1.length, trainTest._1.head._2.length)

      val (min, max) = utils.getMinMax(trainTest._1.map(tup =>
        DenseVector(tup._1.toArray ++ tup._2.toArray)).toList)

      val featuresScaler = new GaussianScaler(min(0 until num_features), max(0 until num_features))

      val targetsScaler = new MinMaxScaler(
        min(num_features until num_features + num_targets),
        max(num_features until num_features + num_targets))

      val scaler: ReversibleScaler[(DenseVector[Double], DenseVector[Double])] = featuresScaler * targetsScaler

      (scaler(trainTest._1), scaler(trainTest._2), (featuresScaler, targetsScaler))
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
    * @return a [[io.github.mandar2812.dynaml.pipes.ParallelPipe]] object.
    * */
  def duplicate[Source, Destination](pipe: DataPipe[Source, Destination]) =
    DataPipe(pipe, pipe)

  /**
    * Constructs a data pipe which performs discrete Haar wavelet transform
    * on a (breeze) vector signal.
    * */
  val haarWaveletFilter = (order: Int) => new ReversibleScaler[DenseVector[Double]] {

    override val i = invHaarWaveletFilter(order)

    override def run(signal: DenseVector[Double]) = {
      //Check size of signal before constructing DWT matrix
      assert(
        signal.length == math.pow(2.0, order).toInt,
        "Signal: "+signal+"\n is of length "+signal.length+
          "\nLength of signal must be : 2^"+order
      )

      // Now construct DWT matrix
      val invSqrtTwo = 1.0/math.sqrt(2.0)

      val rowFactors = (0 until order).reverse.map(i => {
        (1 to math.pow(2.0, i).toInt).map(k =>
          invSqrtTwo/math.sqrt(order-i))})
        .reduceLeft((a,b) => a ++ b).reverse

      val appRowFactors = Seq(rowFactors.head) ++ rowFactors

      val dwtvec = utils.haarMatrix(math.pow(2.0, order).toInt)*signal

      dwtvec.mapPairs((row, v) => v*appRowFactors(row))
    }
  }

  /**
    * Implements the inverse Discrete Haar Wavelet Transform
    *
    * */
  val invHaarWaveletFilter = (order: Int) => Scaler((signal: DenseVector[Double]) => {
    //Check size of signal before constructing DWT matrix
    assert(
      signal.length == math.pow(2.0, order).toInt,
      "Signal: "+signal+"\n is of length "+signal.length+
        "\nLength of signal must be : 2^"+order
    )

    // Now construct DWT matrix
    val invSqrtTwo = 1.0/math.sqrt(2.0)

    val rowFactors = (0 until order).reverse.map(i => {
      (1 to math.pow(2.0, i).toInt).map(k =>
        invSqrtTwo/math.sqrt(order-i))})
      .reduceLeft((a,b) => a ++ b).reverse

    val appRowFactors = Seq(rowFactors.head) ++ rowFactors
    val normalizationMat: DenseMatrix[Double] = diag(DenseVector(appRowFactors.toArray))

    utils.haarMatrix(math.pow(2.0, order).toInt).t*(normalizationMat*signal)
  })

  def trainParametricModel[
  G, T, Q, R, S, M <: ParameterizedLearner[G, T, Q, R, S]
  ](regParameter: Double, step: Double = 0.05,
    maxIt: Int = 50, mini: Double = 1.0) = DataPipe((model: M) => {
      model.setLearningRate(step)
        .setMaxIterations(maxIt)
        .setBatchFraction(mini)
        .setRegParam(regParameter)
        .learn()
      model
    })


  def modelTuning[M <: GloballyOptWithGrad](startingState: Map[String, Double],
                                            globalOpt: String = "GS",
                                            grid: Int = 3, step: Double = 0.02) =
    DataPipe((model: M) => {
      val gs = globalOpt match {
        case "GS" => new GridSearch[M](model)
          .setGridSize(grid)
          .setStepSize(step)
          .setLogScale(false)

        case "ML" => new GPMLOptimizer[DenseVector[Double],
          Seq[(DenseVector[Double], Double)], M](model)

        case "CSA" => new CoupledSimulatedAnnealing(model)
          .setGridSize(grid)
          .setStepSize(step)
          .setLogScale(false)
      }

      gs.optimize(startingState, Map("tolerance" -> "0.0001",
        "step" -> step.toString,
        "maxIterations" -> grid.toString))
    })


  def GPRegressionTest[T <: AbstractGPRegressionModel[
    Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](model:T) =
    DataPipe(
    (trainTest: (Stream[(DenseVector[Double], Double)],
      (DenseVector[Double], DenseVector[Double]))) => {

      val res = model.test(trainTest._1)
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
