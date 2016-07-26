package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.Gaussian
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import org.apache.log4j.Logger

/**
  * @author mandar2812 date: 19/7/16.
  *
  * Abstract implementation of multi-output gaussian process
  * as outlined in Lawrence et. al 2012 on arxiv
  *
  * @tparam T The data structure which is the type of the training
  *           and test data sets.
  *
  * @tparam I The index set of the GP.
  */
abstract class MOGPRegressionModel[T, I](
  cov: CovarianceFunction[I, DenseMatrix[Double], DenseMatrix[Double]],
  n: CovarianceFunction[I, DenseVector[Double], DenseMatrix[Double]],
  data: T, num: Int, numOutputs: Int) extends
  GaussianProcessModel[T, I, DenseVector[Double],
    DenseMatrix[Double], DenseMatrix[Double],
    (DenseVector[Double], DenseMatrix[Double])] {

  private val logger = Logger.getLogger(this.getClass)


  val npoints = num

  val noutputs = numOutputs

  /**
    * The GP is taken to be zero mean, or centered.
    * This is ensured by standardization of the data
    * before being used for further processing.
    *
    * */
  override val mean: (I) => DenseVector[Double] = _ => DenseVector.zeros[Double](noutputs)

  override val covariance = cov

  val noiseModel: CovarianceFunction[I, DenseVector[Double], DenseMatrix[Double]] = n

  override protected val g: T = data

  protected var (caching, kernelMatrixCache, noiseCache)
  : (Boolean, DenseMatrix[Double], DenseMatrix[Double]) = (false, null, null)

  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {
    covariance.setHyperParameters(s)
    noiseModel.setHyperParameters(s)
    //current_state = covariance.state ++ noiseModel.state
    this
  }

  /*override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters

  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state*/

  /**
    * Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U):
  (DenseVector[Double], DenseMatrix[Double]) = {

    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data
    val training = dataAsIndexSeq(g)
    val trainingLabels = dataAsSeq(g).map(_._2).reduce((v1, v2) => DenseVector.vertcat(v1, v2))

    val kernelTraining = if(!caching)
      covariance.buildKernelMatrix(training, npoints).getKernelMatrix()
    else
      kernelMatrixCache

    val noiseMat = if(!caching)
      noiseModel.buildKernelMatrix(training, npoints).getKernelMatrix()
    else
      noiseCache

    val kernelTest = covariance.buildKernelMatrix(test, test.length)
      .getKernelMatrix()
    val crossKernel = covariance.buildCrossKernelMatrix(training, test)

    //Calculate the predictive mean and co-variance
    val inverse = inv(kernelTraining + noiseMat)
    (crossKernel.t * (inverse * trainingLabels),
      kernelTest - (crossKernel.t * (inverse * crossKernel)))
  }

}
