/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 * */
package io.github.mandar2812.dynaml

import scala.collection.mutable.{MutableList => ML}
import breeze.linalg.eig.Eig
import breeze.linalg.{DenseMatrix, DenseVector, eig}
import breeze.numerics.sqrt
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.ParameterizedLearner
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.models.sgp.ESGPModel
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.{
  ContinuousDistrRV,
  ContinuousRVWithDistr
}
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.wavelets.{
  GroupedHaarWaveletFilter,
  HaarWaveletFilter,
  InvGroupedHaarWaveletFilter,
  InverseHaarWaveletFilter
}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.renjin.script.RenjinScriptEngine
import org.renjin.sexp._

//import scalaxy.streams.optimize
import scala.reflect.ClassTag
import scala.util.Random

/**
  * @author mandar2812 datum 3/2/16.
  *
  * A library of sorts for common data processing
  * pipes.
  */
object DynaMLPipe {

  val logger = Logger.getLogger(this.getClass)

  /**
    * A trivial identity data pipe
    * */
  def identityPipe[T] = DataPipe(identity[T] _)

  val unzip: UnzipIterable.type = UnzipIterable

  val tup2_1: Tuple2_1.type = Tuple2_1

  val tup2_2: Tuple2_2.type = Tuple2_2

  /**
    * Data pipe which takes a file name/path as a
    * [[String]] and returns a [[Stream]] of [[String]].
    * */
  val fileToStream = DataPipe(utils.textFileToStream _)

  /**
    * Read a csv text file and store it in a R data frame.
    * @param df The name of the data frame variable
    * @param sep Separation character in the csv file
    * @return A [[DataPipe]] instance which takes as input a file name
    *         and returns a renjin [[ListVector]] instance and stores data frame
    *         in the variable nameed as df.
    * */
  def csvToRDF(
    df: String,
    sep: Char
  )(
    implicit renjin: RenjinScriptEngine
  ): DataPipe[String, ListVector] =
    DataPipe(
      (file: String) =>
        renjin
          .eval(
            df + """ <- read.csv("""" + file + """", sep = '""" + sep + """')"""
          )
          .asInstanceOf[ListVector]
    )

  /**
    * Create a linear model from a R data frame.
    * @param modelName The name of the variable to store model
    * @param y The name of the target variable
    * @param xs A list of names denoting input variables
    * @return A [[DataPipe]] which takes as input data frame variable name
    *         and returns a [[ListVector]] containing linear model attributes.
    *         Also stores the model in the variable given by modelName in the ongoing
    *         R session.
    * */
  def rdfToGLM(
    modelName: String,
    y: String,
    xs: Array[String]
  )(
    implicit renjin: RenjinScriptEngine
  ): DataPipe[String, ListVector] =
    DataPipe(
      (df: String) =>
        renjin
          .eval(
            modelName + " <- lm(" + y + " ~ " + xs
              .mkString(" + ") + ", " + df + ")"
          )
          .asInstanceOf[ListVector]
    )

  /**
    * Writes a [[Stream]] of [[String]] to
    * a file.
    *
    * Usage: DynaMLPipe.streamToFile("abc.csv")
    * */
  val streamToFile = (fileName: String) =>
    DataPipe(utils.writeToFile(fileName) _)

  /**
    * Writes a [[Stream]] of [[AnyVal]] to
    * a file.
    *
    * Usage: DynaMLPipe.valuesToFile("abc.csv")
    * */
  val valuesToFile = (fileName: String) =>
    DataPipe(
      (stream: Stream[Seq[AnyVal]]) =>
        utils.writeToFile(fileName)(stream.map(s => s.mkString(",")))
    )

  /**
    * Drop the first element of a [[Stream]] of [[String]]
    * */
  val dropHead = DataPipe((s: Iterable[String]) => s.tail)

  /**
    * Data pipe to replace all occurrences of a regular expression or string in a [[Stream]]
    * of [[String]] with with a specified replacement string.
    * */
  val replace = (original: String, newString: String) =>
  IterableDataPipe((s: String) => utils.replace(original)(newString)(s))

  /**
    * Data pipe to replace all white spaces in a [[Stream]]
    * of [[String]] with the comma character.
    * */
  val replaceWhiteSpaces = replace("\\s+", ",")

  /**
    * Trim white spaces from each line in a [[Stream]]
    * of [[String]]
    * */
  val trimLines = IterableDataPipe((s: String) => s.trim())

  val splitLine = IterableDataPipe((s: String) => s.split(","))

  /**
    * Generate a numeric range by dividing an interval into bins.
    * */
  val numeric_range: MetaPipe21[Double, Double, Int, Seq[Double]] = MetaPipe21(
    (lower: Double, upper: Double) =>
      (bins: Int) =>
        Seq.tabulate[Double](bins + 1)(
          i =>
            if (i == 0) lower
            else if (i == bins) upper
            else lower + i * (upper - lower) / bins
        )
  )

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
    DataPipe(
      (lines: Iterable[String]) =>
        lines.map { line =>
          val splits = line.split(",")
          val timestamp =
            Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
          (timestamp, splits(3).toDouble)
        }
    )

  /**
    * This pipe is exactly similar to [[DynaMLPipe.extractTimeSeries]],
    * with one key difference, it returns a [[Tuple2]] like
    * (Timestamp, FeatureVector), where FeatureVector is
    * a Vector of values.
    * */
  val extractTimeSeriesVec = (Tfunc: (Double, Double, Double) => Double) =>
    DataPipe(
      (lines: Iterable[String]) =>
        lines.map { line =>
          val splits = line.split(",")
          val timestamp =
            Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble)
          val feat = DenseVector(splits.slice(3, splits.length).map(_.toDouble))
          (timestamp, feat)
        }
    )

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
    DataPipe(
      (lines: Iterable[(Double, Double)]) =>
        lines.toList
          .sliding(deltaT + timelag + 1)
          .map((history) => {
            val features = DenseVector(history.take(deltaT).map(_._2).toArray)
            (features, history.last._2)
          })
          .toStream
    )

  /**
    * The vector version of [[DynaMLPipe.deltaOperation]]
    * */
  val deltaOperationVec = (deltaT: Int) =>
    DataPipe(
      (lines: Iterable[(Double, DenseVector[Double])]) =>
        lines.toList
          .sliding(deltaT + 1)
          .map((history) => {
            val hist                    = history.take(history.length - 1).map(_._2)
            val featuresAcc: ML[Double] = ML()

            (0 until hist.head.length).foreach((dimension) => {
              //for each dimension/regressor take points t to t-order
              featuresAcc ++= hist.map(vec => vec(dimension))
            })

            val features = DenseVector(featuresAcc.toArray)
            (features, history.last._2(0))
          })
          .toStream
    )

  /**
    * The vector ARX version of [[DynaMLPipe.deltaOperation]]
    * */
  val deltaOperationARX = (deltaT: List[Int]) =>
    DataPipe(
      (lines: Iterable[(Double, DenseVector[Double])]) =>
        lines.toList
          .sliding(deltaT.max + 1)
          .map((history) => {
            val hist                    = history.take(history.length - 1).map(_._2)
            val featuresAcc: ML[Double] = ML()

            (0 until hist.head.length).foreach((dimension) => {
              //for each dimension/regressor take points t to t-order
              featuresAcc ++= hist
                .takeRight(deltaT(dimension))
                .map(vec => vec(dimension))
            })

            val features = DenseVector(featuresAcc.toArray)
            (features, history.last._2(0))
          })
          .toStream
    )

  /**
    * From a [[Stream]] of [[String]] remove all records
    * which contain missing values, this pipe should be applied
    * after the application of [[DynaMLPipe.extractTrainingFeatures]].
    * */
  val removeMissingLines = IterableDataPipe(
    (line: String) => !line.contains("<NA>")
  )

  /**
    * Take each line which is a comma separated string and extract
    * all but the last element into a feature vector and leave the last
    * element as the "target" value.
    *
    * This pipe outputs data in a [[Stream]] of [[Tuple2]] in the following form
    *
    * (Vector(features), value)
    * */
  val splitFeaturesAndTargets = IterableDataPipe((line: String) => {
    val split = line.split(",")
    (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
  })

  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  @deprecated(
    "*Standardization pipes are deprecated as of v1.4," +
      " use pipes that output io.github.mandar2812.dynaml.pipes.Scaler objects instead"
  )
  val trainTestGaussianStandardization: DataPipe[
    (
      Iterable[(DenseVector[Double], Double)],
      Iterable[(DenseVector[Double], Double)]
    ),
    (
      (
        Iterable[(DenseVector[Double], Double)],
        Iterable[(DenseVector[Double], Double)]
      ),
      (DenseVector[Double], DenseVector[Double])
    )
  ] =
    DataPipe(
      (trainTest: (
        Iterable[(DenseVector[Double], Double)],
        Iterable[(DenseVector[Double], Double)]
      )) => {

        val (mean, variance) = utils.getStats(
          trainTest._1
            .map(tup => DenseVector(tup._1.toArray ++ Array(tup._2)))
            .toList
        )

        val stdDev: DenseVector[Double] = sqrt(variance)

        val normalizationFunc = (point: (DenseVector[Double], Double)) => {
          val extendedpoint = DenseVector(point._1.toArray ++ Array(point._2))

          val normPoint = (extendedpoint - mean) :/ stdDev
          val length    = normPoint.length
          (normPoint(0 until length - 1), normPoint(-1))
        }

        (
          (
            trainTest._1.map(normalizationFunc),
            trainTest._2.map(normalizationFunc)
          ),
          (mean, stdDev)
        )
      }
    )

  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  @deprecated(
    "*Standardization pipes are deprecated as of v1.4," +
      " use pipes that output io.github.mandar2812.dynaml.pipes.Scaler objects instead"
  )
  val featuresGaussianStandardization: DataPipe[
    (
      Iterable[(DenseVector[Double], Double)],
      Iterable[(DenseVector[Double], Double)]
    ),
    (
      (
        Iterable[(DenseVector[Double], Double)],
        Iterable[(DenseVector[Double], Double)]
      ),
      (DenseVector[Double], DenseVector[Double])
    )
  ] =
    DataPipe(
      (trainTest: (
        Iterable[(DenseVector[Double], Double)],
        Iterable[(DenseVector[Double], Double)]
      )) => {

        val (mean, variance) =
          utils.getStats(trainTest._1.map(tup => tup._1).toList)

        val stdDev: DenseVector[Double] = sqrt(variance)

        val normalizationFunc = (point: (DenseVector[Double], Double)) => {
          val normPoint = (point._1 - mean) :/ stdDev
          (normPoint, point._2)
        }

        (
          (
            trainTest._1.map(normalizationFunc),
            trainTest._2.map(normalizationFunc)
          ),
          (mean, stdDev)
        )
      }
    )

  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  @deprecated(
    "*Standardization pipes are deprecated as of v1.4," +
      " use pipes that output io.github.mandar2812.dynaml.pipes.Scaler objects instead"
  )
  val trainTestGaussianStandardizationMO: DataPipe[
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      Iterable[(DenseVector[Double], DenseVector[Double])]
    ),
    (
      (
        Iterable[(DenseVector[Double], DenseVector[Double])],
        Iterable[(DenseVector[Double], DenseVector[Double])]
      ),
      (DenseVector[Double], DenseVector[Double])
    )
  ] =
    DataPipe(
      (trainTest: (
        Iterable[(DenseVector[Double], DenseVector[Double])],
        Iterable[(DenseVector[Double], DenseVector[Double])]
      )) => {

        val (mean, variance) = utils.getStats(
          trainTest._1
            .map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray))
            .toList
        )

        val stdDev: DenseVector[Double] = sqrt(variance)

        val normalizationFunc =
          (point: (DenseVector[Double], DenseVector[Double])) => {
            val extendedpoint =
              DenseVector(point._1.toArray ++ point._2.toArray)

            val normPoint = (extendedpoint - mean) :/ stdDev
            val length    = point._1.length
            val outlength = point._2.length

            (
              normPoint(0 until length),
              normPoint(length until length + outlength)
            )
          }

        (
          (
            trainTest._1.map(normalizationFunc),
            trainTest._2.map(normalizationFunc)
          ),
          (mean, stdDev)
        )
      }
    )

  /**
    * Returns a pipe which takes a data set and calculates the mean and standard deviation of each dimension.
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[GaussianScaler]] instances.
    * */
  def calculateGaussianScales(standardize: Boolean = true): DataPipe[Iterable[
    (DenseVector[Double], DenseVector[Double])
  ], (Iterable[(DenseVector[Double], DenseVector[Double])], (GaussianScaler, GaussianScaler))] =
    DataPipe((data: Iterable[(DenseVector[Double], DenseVector[Double])]) => {

      val (num_features, num_targets) =
        (data.head._1.length, data.head._2.length)

      val (mean, variance) = utils.getStats(
        data.map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray)).toList
      )

      val stdDev: DenseVector[Double] = sqrt(variance)

      val featuresScaler =
        GaussianScaler(mean(0 until num_features), stdDev(0 until num_features))

      val targetsScaler = GaussianScaler(
        mean(num_features until num_features + num_targets),
        stdDev(num_features until num_features + num_targets)
      )

      val result =
        if (standardize) (featuresScaler * targetsScaler)(data) else data

      (result, (featuresScaler, targetsScaler))
    })

  /**
    * Returns a pipe which takes a data set and mean centers it.
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[MeanScaler]] instances.
    * */
  def calculateMeanScales(standardize: Boolean = true): DataPipe[Iterable[
    (DenseVector[Double], DenseVector[Double])
  ], (Iterable[(DenseVector[Double], DenseVector[Double])], (MeanScaler, MeanScaler))] =
    DataPipe((data: Iterable[(DenseVector[Double], DenseVector[Double])]) => {

      val (num_features, num_targets) =
        (data.head._1.length, data.head._2.length)

      val (mean, _) = utils.getStats(
        data.map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray)).toList
      )

      val featuresScaler = MeanScaler(mean(0 until num_features))

      val targetsScaler =
        MeanScaler(mean(num_features until num_features + num_targets))

      val result =
        if (standardize) (featuresScaler * targetsScaler)(data) else data

      (result, (featuresScaler, targetsScaler))
    })

  /**
    * Multivariate version of [[calculateGaussianScales]]
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[MVGaussianScaler]] instances.
    * */
  def calculateMVGaussianScales(standardize: Boolean = true): DataPipe[Iterable[
    (DenseVector[Double], DenseVector[Double])
  ], (Iterable[(DenseVector[Double], DenseVector[Double])], (MVGaussianScaler, MVGaussianScaler))] =
    DataPipe((data: Iterable[(DenseVector[Double], DenseVector[Double])]) => {

      val (num_features, num_targets) =
        (data.head._1.length, data.head._2.length)

      val (m, sigma) = utils.getStatsMult(
        data.map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray)).toList
      )

      val featuresScaler = MVGaussianScaler(
        m(0 until num_features),
        sigma(0 until num_features, 0 until num_features)
      )

      val targetsScaler = MVGaussianScaler(
        m(num_features until num_features + num_targets),
        sigma(
          num_features until num_features + num_targets,
          num_features until num_features + num_targets
        )
      )

      val result =
        if (standardize) (featuresScaler * targetsScaler)(data) else data

      (result, (featuresScaler, targetsScaler))
    })

  /**
    * Returns a pipe which performs PCA on data features and gaussian scaling on data targets
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[MVGaussianScaler]] instances.
    * */
  def calculatePCAScales(standardize: Boolean = true): DataPipe[
    Iterable[(DenseVector[Double], DenseVector[Double])],
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (PCAScaler, MVGaussianScaler)
    )
  ] =
    DataPipe((data: Iterable[(DenseVector[Double], DenseVector[Double])]) => {

      val (num_features, num_targets) =
        (data.head._1.length, data.head._2.length)

      val (m, sigma) = utils.getStatsMult(
        data.map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray)).toList
      )

      val Eig(eigenvalues, _, eigenvectors) = eig(sigma(0 until num_features, 0 until num_features))

      val featuresScaler = PCAScaler(
        m(0 until num_features),
        eigenvalues,
        eigenvectors
      )

      val targetsScaler = MVGaussianScaler(
        m(num_features until num_features + num_targets),
        sigma(
          num_features until num_features + num_targets,
          num_features until num_features + num_targets
        )
      )

      val result =
        if (standardize) (featuresScaler * targetsScaler)(data) else data

      (result, (featuresScaler, targetsScaler))
    })

  /**
    * Returns a pipe which performs PCA on data features and gaussian scaling on data targets
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[MVGaussianScaler]] instances.
    * */
  def calculatePCAScalesFeatures(standardize: Boolean = true): DataPipe[
    Iterable[DenseVector[Double]],
    (Iterable[DenseVector[Double]], PCAScaler)
  ] =
    DataPipe((data: Iterable[DenseVector[Double]]) => {

      val (m, sigma) = utils.getStatsMult(data.toList)

      val Eig(eigenvalues, _, eigenvectors) = eig(sigma)

      val featuresScaler = PCAScaler(m, eigenvalues, eigenvectors)

      val result = if (standardize) featuresScaler(data) else data

      (result, featuresScaler)
    })

  /**
    * Returns a pipe which takes a data set and calculates the minimum and maximum of each dimension.
    * @param standardize Set to true if one wants the standardized data and false if one
    *                    does wants the original data with the [[MinMaxScaler]] instances.
    * */
  def calculateMinMaxScales(standardize: Boolean = true): DataPipe[
    Iterable[(DenseVector[Double], DenseVector[Double])],
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (MinMaxScaler, MinMaxScaler)
    )
  ] =
    DataPipe((data: Iterable[(DenseVector[Double], DenseVector[Double])]) => {

      val (num_features, num_targets) =
        (data.head._1.length, data.head._2.length)

      val (min, max) = utils.getMinMax(
        data.map(tup => DenseVector(tup._1.toArray ++ tup._2.toArray)).toList
      )

      val featuresScaler =
        MinMaxScaler(min(0 until num_features), max(0 until num_features))

      val targetsScaler = MinMaxScaler(
        min(num_features until num_features + num_targets),
        max(num_features until num_features + num_targets)
      )

      val result =
        if (standardize) (featuresScaler * targetsScaler)(data) else data

      (result, (featuresScaler, targetsScaler))
    })

  /**
    * A helper method which takes a scaled data set and applies its scales to
    * a test set.
    * */
  private[dynaml] def scaleTestPipe[I, R <: ReversibleScaler[I]] = DataPipe(
    (couple: ((Iterable[(I, I)], (R, R)), Iterable[(I, I)])) =>
      (
        couple._1._1,
        (couple._1._2._1 * couple._1._2._2)(couple._2),
        couple._1._2
      )
  )

  /**
    * Scale a data set which is stored as a [[Stream]],
    * return the scaled data as well as a [[GaussianScaler]] instance
    * which can be used to reverse the scaled values to the original
    * data.
    *
    * */
  val gaussianScaling: DataPipe[
    Iterable[(DenseVector[Double], DenseVector[Double])],
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )
  ] =
    calculateGaussianScales()

  /**
    * Scale a data set which is stored as a [[Stream]],
    * return the scaled data as well as a [[MVGaussianScaler]] instance
    * which can be used to reverse the scaled values to the original
    * data.
    * */
  val multivariateGaussianScaling: DataPipe[
    Iterable[(DenseVector[Double], DenseVector[Double])],
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (MVGaussianScaler, MVGaussianScaler)
    )
  ] =
    calculateMVGaussianScales()

  /**
    * Perform gaussian normalization on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  val gaussianScalingTrainTest: DataPipe[
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      Iterable[(DenseVector[Double], DenseVector[Double])]
    ),
    (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )
  ] =
    (calculateGaussianScales() * identityPipe[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ]]) >
      scaleTestPipe[DenseVector[Double], GaussianScaler]

  /**
    * Scale a data set which is stored as a [[Stream]],
    * return the scaled data as well as a [[MVGaussianScaler]] instance
    * which can be used to reverse the scaled values to the original
    * data.
    * */
  val multivariateGaussianScalingTrainTest =
    (
      calculateMVGaussianScales() *
        identityPipe[Iterable[(DenseVector[Double], DenseVector[Double])]]
    ) >
      scaleTestPipe[DenseVector[Double], MVGaussianScaler]

  /**
    * Transform a data set by performing PCA on its patterns.
    * */
  val pcaFeatureScaling = calculatePCAScalesFeatures()

  /**
    * Transform a data set consisting of features and targets.
    * Perform PCA scaling of features and gaussian scaling of targets.
    * */
  val pcaScaling = calculatePCAScales()

  /**
    * Scale a data set which is stored as a [[Stream]],
    * return the scaled data as well as a [[MinMaxScaler]] instance
    * which can be used to reverse the scaled values to the original
    * data.
    *
    * */
  val minMaxScaling: DataPipe[Iterable[
    (DenseVector[Double], DenseVector[Double])
  ], (Iterable[(DenseVector[Double], DenseVector[Double])], (MinMaxScaler, MinMaxScaler))] =
    calculateMinMaxScales()

  /**
    * Perform [0,1] scaling on a data stream which
    * is a [[Tuple2]] of the form.
    *
    * (Stream(training data), Stream(test data))
    * */
  val minMaxScalingTrainTest =
    (calculateMinMaxScales() * identityPipe[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ]]) >
      scaleTestPipe[DenseVector[Double], MinMaxScaler]

  /**
    * Extract a subset of the data into a [[Tuple2]] which
    * can be used as a training, test combo for model learning and evaluation.
    *
    * Usage: DynaMLPipe.splitTrainingTest(num_training, num_test)
    * */
  def splitTrainingTest[P](num_training: Int, num_test: Int) =
    DataPipe((data: (Iterable[P], Iterable[P])) => {
      (data._1.take(num_training), data._2.takeRight(num_test))
    })

  /**
    * Extract a subset of columns from a [[Stream]] of comma separated [[String]]
    * also replace any missing value strings with the empty string.
    *
    * Usage: DynaMLPipe.extractTrainingFeatures(List(1,2,3), Map(1 -> "N.A.", 2 -> "NA", 3 -> "na"))
    * */
  val extractTrainingFeatures =
    (columns: List[Int], m: Map[Int, String]) =>
      DataPipe((l: Iterable[String]) => utils.extractColumns(l, ",", columns, m))

  /**
    * Returns a pipeline which performs a bagging based sub-sampling
    * of a stream of [[T]].
    *
    * @param proportion The sampling proportion between 0 and 1
    * @param nBags The number of bags to generate.
    * */
  def baggingIterable[T](proportion: Double, nBags: Int) = {
    require(
      proportion > 0.0 && proportion <= 1.0 && nBags > 0,
      "Sampling proprotion must be between 0 and 1; " +
        "Number of bags must be positive"
    )
    DataPipe((data: Stream[T]) => {
      val data_size = data.toSeq.length
      val sizeOfBag: Int = (data_size * proportion).toInt
      (1 to nBags)
        .map(
          _ =>
            Stream
              .tabulate[T](sizeOfBag)(_ => data(Random.nextInt(data_size)))
        )
        .toStream
    })
  }

  /**
    * Returns a pipeline which performs a bagging based sub-sampling
    * of an Apache Spark [[RDD]] of [[T]].
    *
    * @param proportion The sampling proportion between 0 and 1
    * @param nBags The number of bags to generate.
    * */
  def baggingRDD[T](proportion: Double, nBags: Int) = {
    require(
      proportion > 0.0 && proportion <= 1.0 && nBags > 0,
      "Sampling proprotion must be between 0 and 1; " +
        "Number of bags must be positive"
    )
    DataPipe(
      (data: RDD[T]) =>
        (1 to nBags).map(_ => data.sample(withReplacement = true, proportion))
    )
  }

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
  val haarWaveletFilter = (order: Int) => HaarWaveletFilter(order)

  /**
    * Constructs a data pipe which performs inverse discrete Haar wavelet transform
    * on a (breeze) vector signal.
    * */
  val invHaarWaveletFilter = (order: Int) => InverseHaarWaveletFilter(order)

  val groupedHaarWaveletFilter = (orders: Array[Int]) =>
    GroupedHaarWaveletFilter(orders)

  val invGroupedHaarWaveletFilter = (orders: Array[Int]) =>
    InvGroupedHaarWaveletFilter(orders)

  def genericReplicationEncoder[I](
    n: Int
  )(
    implicit tag: ClassTag[I]
  ): Encoder[I, Array[I]] =
    Encoder[I, Array[I]]((v: I) => {
      Array.fill[I](n)(v)
    }, (vs: Array[I]) => {
      vs.head
    })

  /**
    * Creates an [[Encoder]] which can split
    * [[DenseVector]] instances into uniform splits and
    * put them back together.
    * */
  val breezeDVSplitEncoder = (n: Int) =>
    Encoder(
      (v: DenseVector[Double]) =>
        Array.tabulate(v.length / n)(
          i => v(i * n until math.min((i + 1) * n, v.length))
        ),
      (vs: Array[DenseVector[Double]]) =>
        DenseVector(vs.map(_.toArray).reduceLeft((a, b) => a ++ b))
    )

  /**
    * Creates an [[Encoder]] which replicates a
    * [[DenseVector]] instance n times.
    * */
  val breezeDVReplicationEncoder = (n: Int) =>
    genericReplicationEncoder[DenseVector[Double]](n)

  def trainParametricModel[
    G,
    T,
    Q,
    R,
    S,
    M <: ParameterizedLearner[G, T, Q, R, S]
  ](regParameter: Double,
    step: Double = 0.05,
    maxIt: Int = 50,
    mini: Double = 1.0
  ) =
    DataPipe((model: M) => {
      model
        .setLearningRate(step)
        .setMaxIterations(maxIt)
        .setBatchFraction(mini)
        .setRegParam(regParameter)
        .learn()
      model
    })

  def modelTuning[M <: GloballyOptWithGrad](
    startingState: Map[String, Double],
    globalOpt: String = "GS",
    grid: Int = 3,
    step: Double = 0.02
  ) =
    DataPipe((model: M) => {
      val gs = globalOpt match {
        case "GS" =>
          new GridSearch[M](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)

        case "ML" => new GradBasedGlobalOptimizer[M](model)

        case "CSA" =>
          new CoupledSimulatedAnnealing(model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)
            .setVariant(AbstractCSA.MwVC)
      }

      gs.optimize(
        startingState,
        Map(
          "tolerance"     -> "0.0001",
          "step"          -> step.toString,
          "maxIterations" -> grid.toString
        )
      )
    })

  def gpTuning[T, I: ClassTag](
    startingState: Map[String, Double],
    globalOpt: String = "GS",
    grid: Int = 3,
    step: Double = 0.02,
    maxIt: Int = 20,
    policy: String = "GS",
    prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] =
      Map()
  ) =
    DataPipe((model: AbstractGPRegressionModel[T, I]) => {
      val gs = globalOpt match {
        case "GS" =>
          new GridSearch(model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)
            .setPrior(prior)
            .setNumSamples(prior.size * grid)

        case "ML" => new GradBasedGlobalOptimizer(model)

        case "CSA" =>
          new CoupledSimulatedAnnealing(model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
            .setPrior(prior)
            .setNumSamples(prior.size * grid)

        case "GPC" =>
          new ProbGPCommMachine(model)
            .setPolicy(policy)
            .setGridSize(grid)
            .setStepSize(step)
            .setMaxIterations(maxIt)
            .setPrior(prior)
            .setNumSamples(prior.size * grid)
      }

      gs.optimize(
        startingState,
        Map(
          "tolerance"     -> "0.0001",
          "step"          -> step.toString,
          "maxIterations" -> grid.toString,
          "persist"       -> "true"
        )
      )
    })

  def sgpTuning[T, I: ClassTag](
    startingState: Map[String, Double],
    globalOpt: String = "GS",
    grid: Int = 3,
    step: Double = 0.02,
    maxIt: Int = 20,
    prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] =
      Map()
  ) =
    DataPipe((model: ESGPModel[T, I]) => {
      val gs = globalOpt match {
        case "GS" =>
          new GridSearch(model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)
            .setPrior(prior)
            .setNumSamples(prior.size * grid)

        case "CSA" =>
          new CoupledSimulatedAnnealing(model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
            .setPrior(prior)
            .setNumSamples(prior.size * grid)

      }

      gs.optimize(
        startingState,
        Map(
          "tolerance"     -> "0.0001",
          "step"          -> step.toString,
          "maxIterations" -> grid.toString,
          "persist"       -> "true"
        )
      )
    })

  def GPRegressionTest[
    T <: AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)], DenseVector[
      Double
    ]]
  ](model: T
  ) =
    DataPipe(
      (trainTest: (
        Iterable[(DenseVector[Double], Double)],
        (DenseVector[Double], DenseVector[Double])
      )) => {

        val res = model.test(trainTest._1.toSeq)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2)).toList
          ) > DataPipe(
            (list: List[(Double, Double)]) =>
              list.map { l =>
                (
                  l._1 * trainTest._2._2(-1) + trainTest._2._1(-1),
                  l._2 * trainTest._2._2(-1) + trainTest._2._1(-1)
                )
              }
          )

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics =
          new RegressionMetrics(scoresAndLabels, scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()
      }
    )
}
