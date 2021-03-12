package io.github.tailhq.dynaml.wavelets

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import io.github.tailhq.dynaml.pipes.{ReversibleScaler, Scaler}
import io.github.tailhq.dynaml.utils

/**
  * @author tailhq date: 14/12/2016.
  */
case class HaarWaveletFilter(order: Int) extends ReversibleScaler[DenseVector[Double]] {

  val invSqrtTwo = 1.0/math.sqrt(2.0)

  val rowFactors = (0 until order).reverse.map(i => {
    (1 to math.pow(2.0, i).toInt).map(k =>
      invSqrtTwo/math.sqrt(order-i))})
    .reduceLeft((a,b) => a ++ b).reverse

  val appRowFactors = Seq(rowFactors.head) ++ rowFactors

  lazy val normalizationMat: DenseMatrix[Double] = diag(DenseVector(appRowFactors.toArray))

  lazy val transformMat: DenseMatrix[Double] = utils.haarMatrix(math.pow(2.0, order).toInt)

  override val i = InverseHaarWaveletFilter(order)

  override def run(signal: DenseVector[Double]) = {
    //Check size of signal before constructing DWT matrix
    assert(
      signal.length == math.pow(2.0, order).toInt,
      "Signal: "+signal+"\n is of length "+signal.length+
        "\nLength of signal must be : 2^"+order
    )
    normalizationMat*(transformMat*signal)
  }
}

case class InverseHaarWaveletFilter(order: Int) extends Scaler[DenseVector[Double]] {

  val invSqrtTwo = 1.0/math.sqrt(2.0)

  val rowFactors = (0 until order).reverse.map(i => {
    (1 to math.pow(2.0, i).toInt).map(k =>
      invSqrtTwo/math.sqrt(order-i))})
    .reduceLeft((a,b) => a ++ b).reverse

  val appRowFactors = Seq(rowFactors.head) ++ rowFactors

  lazy val normalizationMat: DenseMatrix[Double] = diag(DenseVector(appRowFactors.toArray))

  lazy val transformMat: DenseMatrix[Double] = utils.haarMatrix(math.pow(2.0, order).toInt).t

  override def run(signal: DenseVector[Double]): DenseVector[Double] = {
    assert(
      signal.length == math.pow(2.0, order).toInt,
      "Signal: "+signal+"\n is of length "+signal.length+
        "\nLength of signal must be : 2^"+order
    )

    transformMat*(normalizationMat*signal)
  }
}


/**
  * Computes Discrete Wavelet Transform when features are time shifted
  * groups of various quantities. Often this is encountered in NARX models
  * where a feature vector may look like (x_1, x_2, ..., y_1, y_2, ...)
  *
  * The class groups the dimensions into separate vectors for each variable
  * and computes DWT on each group.
  *
  * @param orders A list containing the time exponents of each variable, the auto-regressive
  *               order is 2 exp (order)
  * */
case class GroupedHaarWaveletFilter(orders: Array[Int]) extends ReversibleScaler[DenseVector[Double]] {

  val componentFilters: Array[HaarWaveletFilter] = orders.map(HaarWaveletFilter)

  val twoExp = (i: Int) => math.pow(2.0, i).toInt

  val partitionIndices: Array[(Int, Int)] =
    orders.map(twoExp).scanLeft(0)(_+_).sliding(2).map(c => (c.head, c.last)).toArray

  assert(partitionIndices.length == orders.length, "Number of partitions must be equal to number of variable groups")

  override val i: InvGroupedHaarWaveletFilter = InvGroupedHaarWaveletFilter(orders)

  override def run(data: DenseVector[Double]): DenseVector[Double] = DenseVector(
    partitionIndices.zip(componentFilters).map(limitsAndFilter => {
      val ((start, end), filter) = limitsAndFilter
      filter(data(start until end)).toArray
    }).reduceLeft((a,b) => a ++ b)
  )

}

/**
  * Inverse of the [[GroupedHaarWaveletFilter]]
  *
  * */
case class InvGroupedHaarWaveletFilter(orders: Array[Int]) extends Scaler[DenseVector[Double]] {

  val componentFilters: Array[InverseHaarWaveletFilter] = orders.map(InverseHaarWaveletFilter)

  val twoExp = (i: Int) => math.pow(2.0, i).toInt

  val partitionIndices: Array[(Int, Int)] =
    orders.map(twoExp).scanLeft(0)(_+_).sliding(2).map(c => (c.head, c.last)).toArray

  assert(partitionIndices.length == orders.length, "Number of partitions must be equal to number of variable groups")

  override def run(data: DenseVector[Double]): DenseVector[Double] = DenseVector(
    partitionIndices.zip(componentFilters).map(limitsAndFilter => {
      val ((start, end), filter) = limitsAndFilter
      filter(data(start until end)).toArray
    }).reduceLeft((a,b) => a ++ b)
  )

}