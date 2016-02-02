package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseVector, DenseMatrix}
import com.github.tototoshi.csv.CSVReader
import io.github.mandar2812.dynaml.kernels.CovarianceFunction

/**
  * Created by mandar on 16/11/15.
  */
class GPTimeSeries(cov: CovarianceFunction[Double, Double, DenseMatrix[Double]],
                   n: CovarianceFunction[Double, Double, DenseMatrix[Double]],
                   trainingdata: Seq[(Double, Double)])
  extends AbstractGPRegressionModel[Seq[(Double, Double)],Double](cov, n, trainingdata,
    trainingdata.length){
  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(Double, Double)]) = data
}

object GPTimeSeries {
  def buildFromFile(file : CSVReader, columns: List[Int],
                    missingValues: List[Double], length: Int,
                    processFunc: Seq[String] => (Double, Double),
                    fraction: Double = 0.75):
  (Seq[(Double, Double)], Seq[(Double, Double)]) = {
    assert(columns.length == missingValues.length || missingValues.isEmpty,
      "Missing values must be of equal number to features")
    val stream = file.iterator
    val trLength = (length*fraction).toInt
    val trainingdata = stream.take(trLength).map(processFunc).toSeq
    val testData = stream.toSeq.map(processFunc)
    (trainingdata, testData)
  }


}