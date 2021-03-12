package io.github.tailhq.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.tailhq.dynaml.algebra.{PartitionedMatrix, PartitionedPSDMatrix, PartitionedVector, bcholesky}
import io.github.tailhq.dynaml.algebra.PartitionedMatrixOps._
import io.github.tailhq.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.tailhq.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}
import org.apache.log4j.Logger

/**
  * @author tailhq date: 17/02/2017.
  */
class GeneralizedLeastSquaresModel(
  data: Stream[(DenseVector[Double], Double)],
  Omega: PartitionedPSDMatrix,
  map: (DenseVector[Double]) => DenseVector[Double] =
  identity[DenseVector[Double]]) extends
  GenericGLM[
    (Stream[(DenseVector[Double], Double)], PartitionedPSDMatrix),
    (DenseMatrix[Double], DenseVector[Double])](
    (data, Omega), data.length.toLong) {

  private val logger = Logger.getLogger(this.getClass)

  override def prepareData(d: (Stream[(DenseVector[Double], Double)], PartitionedPSDMatrix)) = {
    logger.info("Performing sanity checks")
    //Sanity checks
    require(d._1.length == Omega.rows, "Sizes of data set and covariance matrix of residuals must match!!")

    //Create design matrix from input features
    //Find out the blockSize for the covariance matrix of residuals Omega
    val bS: Int = (Omega.rows/Omega.rowBlocks).toInt

    logger.info("Constructing blocks of X (input matrix) and y (output vector)")

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
