package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import io.github.mandar2812.dynaml.kernels.{SVMKernel, AbstractKernel}

/**
 * Gaussian Process Regression Model
 * Performs gp/spline smoothing/regression
 * with vector inputs and a singular scalar output.
 *
 *
 */
class GPRegressionModel(cov: SVMKernel[DenseMatrix[Double]], data: FramedGraph[Graph]) extends
GaussianProcessModel[FramedGraph[Graph], DenseVector[Double], Double]{
  override val mean: (DenseVector[Double]) => Double = _ => 0.0

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   **/
  override def predictiveDistribution[U <: Seq[DenseVector[Double]]](test: U): MultivariateGaussian = {

    new MultivariateGaussian(DenseVector(0.0), DenseMatrix(0.0))
  }

  /**
   * Returns a prediction with error bars for a test set.
   **/
  override def test[U <: Seq[(DenseVector[Double], Double)],
                    V <: Seq[(DenseVector[Double], Double,
                      Double, Double, Double)]](test: U): V = ???

  override val covariance: SVMKernel[DenseMatrix[Double]] = cov
  override protected val g: FramedGraph[Graph] = data
}
