package io.github.mandar2812.dynaml.pipes

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

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

  val deltaOperation = (deltaT: Int, timelag: Int) =>
    DataPipe((lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(deltaT+timelag+1).map((history) => {
        val features = DenseVector(history.take(deltaT).map(_._2).toArray)
        (features, history.last._2)
      }).toStream)

  val removeMissingLines = StreamDataPipe((line: String) => !line.contains(",,"))

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

}
