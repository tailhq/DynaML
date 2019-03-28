package io.github.mandar2812.dynaml.utils

import breeze.linalg.eig.Eig
import breeze.linalg.{DenseMatrix, DenseVector, eig}
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * Transforms data by projecting
  * on the principal components (eigen-vectors)
  * of the sample covariance matrix.
  *
  * @param center The empirical mean of the data features
  * @param covmat The empirical covariance matrix of the data features
  * @author mandar2812 date 30/05/2017.
  * */
case class PCAScaler(
  center: DenseVector[Double],
  eigenvalues: DenseVector[Double],
  eigenvectors: DenseMatrix[Double]) extends
  ReversibleScaler[DenseVector[Double]] { self =>

  override val i = Scaler((data: DenseVector[Double]) => (eigenvectors*data)+center)

  override def run(data: DenseVector[Double]) = eigenvectors.t*(data-center)

  def apply(r: Range): CompressedPCAScaler = CompressedPCAScaler(
    r,
    self.center, 
    self.eigenvalues, 
    self.eigenvectors)
}

case class CompressedPCAScaler(
  r: Range,
  center: DenseVector[Double],
  eigenvalues: DenseVector[Double],
  eigenvectors: DenseMatrix[Double]
) extends Scaler[DenseVector[Double]] {

  override def run(data: DenseVector[Double]) = {
    val projections = eigenvectors.t*(data-center)
    projections(r)
  }
}