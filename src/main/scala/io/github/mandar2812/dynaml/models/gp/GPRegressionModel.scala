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
GaussianProcessModel[FramedGraph[Graph], DenseVector[Double], Double, MultivariateGaussian]{

  override val mean: (DenseVector[Double]) => Double = _ => 0.0

  override val covariance: SVMKernel[DenseMatrix[Double]] = cov

  override protected val g: FramedGraph[Graph] = data

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   **/
  override def predictiveDistribution[U <: Seq[DenseVector[Double]]](test: U): MultivariateGaussian = {
    val trainKernelMat
    new MultivariateGaussian(DenseVector(0.0), DenseMatrix(0.0))
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - )
    * 3) Y+ : The upper error bar.
    **/
  override def predictionWithErrorBars[U <: Seq[DenseVector[Double]]](testData: U, confidence: Double):
  Seq[(DenseVector[Double], Double, Double, Double)] = {


    Seq()
  }

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: FramedGraph[Graph]): Seq[(DenseVector[Double], Double)] = ???
}
