package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.algebra.{PartitionedMatrix, PartitionedPSDMatrix, PartitionedVector, bcholesky}
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}

/**
  * @author mandar2812 date: 25/01/2017.
  *
  * The base class for representing single output
  * Generalised Linear Models.
  *
  * See also [[GeneralizedLinearModel]] and [[SparkGLM]]
  */
abstract class GenericGLM[Data, T](
  data: Data, numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]])
  extends LinearModel[Data, DenseVector[Double], DenseVector[Double], Double, T] {

  featureMap = map

  val h: (Double) => Double = identity

  def prepareData(d: Data): T

  override def predict(point: DenseVector[Double]): Double =
    h(params dot DenseVector(featureMap(point).toArray ++ Array(1.0)))

  override def clearParameters(): Unit = {
    params = initParams()
  }

  /**
    * The training data
    **/
  override protected val g: Data = data

  override def learn(): Unit = {
    params = optimizer.optimize(numPoints,
      prepareData(g), initParams())
  }

}


abstract class GeneralizedLeastSquaresModel(
  data: Stream[(DenseVector[Double], Double)],
  Omega: PartitionedPSDMatrix,
  numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] =
  identity[DenseVector[Double]]) extends
  GenericGLM[
    (Stream[(DenseVector[Double], Double)], PartitionedPSDMatrix),
    (DenseMatrix[Double], DenseVector[Double])](
    (data, Omega), data.length.toLong) {

  override def prepareData(d: (Stream[(DenseVector[Double], Double)], PartitionedPSDMatrix)) = {
    //Sanity checks
    require(d._1.length == Omega.rows, "Sizes of data set and covariance matrix of residuals must match!!")

    //Create design matrix from input features
    //Find out the blockSize for the covariance matrix of residuals Omega
    val bS: Int = (Omega.rows/Omega.rowBlocks).toInt

    val (xMatBlocks, yVecBlocks) =
      d._1.grouped(bS).map(blocks => {
        val (localMatBl, localVecBl) = blocks.unzip
        (
          DenseMatrix.vertcat(
            localMatBl.map(features =>
              DenseVector(featureMap(features).toArray ++ Array(1.0)).toDenseMatrix
            ):_*),
          DenseVector(localVecBl.toArray))
      }).toStream.unzip

    val (x, y) = (
      PartitionedMatrix(
        xMatBlocks.zipWithIndex.map(b => ((b._2.toLong, 0L), b._1)),
        numcols = dimensions + 1L, numrows = Omega.rows),
      PartitionedVector(
        yVecBlocks.zipWithIndex.map(d => (d._1, d._2.toLong)).map(_.swap),
        Omega.rows)
    )

    //Perform Cholesky decomposition on Omega
    val lowerT = bcholesky(Omega)

    //Use result of cholesky decomposition to solve linear systems
    //to yield the design matrix and response vector
    val (designMatrix, responseVector): (PartitionedMatrix, PartitionedVector) = (
      x.t * (lowerT.t \\ (lowerT \\ x)),
      x.t * (lowerT.t \\ (lowerT \\ y))
    )

    (designMatrix.toBreezeMatrix, responseVector.toBreezeVector)
  }

  override protected var params: DenseVector[Double] = initParams()

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double], Double,
    (DenseMatrix[Double], DenseVector[Double])] =
    new RegularizedLSSolver().setRegParam(0.0)

  def dimensions = featureMap(data.head._1).length

  /**
    * Initialize parameters to a vector of ones.
    * */
  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](dimensions + 1)
}
