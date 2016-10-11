package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.{DenseMatrix, DenseVector, cholesky}
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.{ContinuousProcess, SecondOrderProcess}
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.probability.MultStudentsTRV
import io.github.mandar2812.dynaml.probability.distributions.MultivariateStudentsT
import org.apache.log4j.Logger

/**
  * Created by mandar on 26/08/16.
  */
abstract class AbstractSTPRegressionModel[T, I](
  mu: Double, cov: LocalScalarKernel[I],
  n: LocalScalarKernel[I],
  data: T, num: Int)
  extends ContinuousProcess[T, I, Double, MultStudentsTRV]
  with SecondOrderProcess[T, I, Double, Double, DenseMatrix[Double], MultStudentsTRV]
  with GloballyOptimizable {


  private val logger = Logger.getLogger(this.getClass)

  /**
    * The GP is taken to be zero mean, or centered.
    * This is ensured by standardization of the data
    * before being used for further processing.
    *
    * */
  override val mean: (I) => Double = _ => 0.0

  override val covariance = cov

  val noiseModel = n

  override protected val g: T = data

  val npoints = num

  protected var (caching, kernelMatrixCache)
  : (Boolean, DenseMatrix[Double]) = (false, null)


  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {
    covariance.setHyperParameters(s)
    noiseModel.setHyperParameters(s)
    current_state = covariance.state ++ noiseModel.state
    current_state += ("degrees_of_freedom" -> (s("degrees_of_freedom")+2.0))
    this
  }

  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters ++ List("degrees_of_freedom")


  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state ++ Map("degrees_of_freedom" -> (2.0+mu))

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: I): Double = predictionWithErrorBars(Seq(point), 1).head._2

  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U): MultStudentsTRV = {
    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data
    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val effectiveTrainingKernel = covariance + noiseModel

    val smoothingMat = if(!caching)
      effectiveTrainingKernel.buildKernelMatrix(training, npoints).getKernelMatrix()
    else
      kernelMatrixCache

    val kernelTest = covariance.buildKernelMatrix(test, test.length)
      .getKernelMatrix()
    val crossKernel = covariance.buildCrossKernelMatrix(training, test)

    //Calculate the predictive mean and co-variance
    //val smoothingMat = kernelTraining + noiseMat
    val Lmat = cholesky(smoothingMat)
    val alpha = Lmat.t \ (Lmat \ trainingLabels)
    val v = Lmat \ crossKernel

    val varianceReducer = v.t * v
    //Ensure that v is symmetric

    val adjustedVarReducer = DenseMatrix.tabulate[Double](varianceReducer.rows, varianceReducer.cols)(
      (i,j) => if(i <= j) varianceReducer(i,j) else varianceReducer(j,i))

    val degOfFreedom = current_state("degrees_of_freedom")

    val beta = trainingLabels dot alpha

    val varianceAdjustment = (degOfFreedom + beta - 2.0)/(degOfFreedom + training.length - 2.0)

    MultStudentsTRV(test.length)(
      training.length+degOfFreedom,
      crossKernel.t * alpha,
      (kernelTest - adjustedVarReducer)*varianceAdjustment)
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int):
  Seq[(I, Double, Double, Double)] = {

    val posterior = predictiveDistribution(testData)
    val postcov = posterior.covariance
    val postmean = posterior.mean
    val stdDev = (1 to testData.length).map(i => math.sqrt(postcov(i-1, i-1)))
    val mean = postmean.toArray.toSeq

    logger.info("Generating error bars")
    val preds = (mean zip stdDev).map(j => (j._1, j._1 - sigma*j._2, j._1 + sigma*j._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    *
    * In this particular case E(h) = -log p(Y|X,h)
    * also known as log likelihood.
    **/
  override def energy(h: Map[String, Double], options: Map[String, String]): Double = {

    setState(h)
    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val effectiveTrainingKernel = covariance + noiseModel

    val kernelTraining: DenseMatrix[Double] =
      effectiveTrainingKernel.buildKernelMatrix(training, npoints).getKernelMatrix()

    AbstractSTPRegressionModel.logLikelihood(current_state("degrees_of_freedom"),trainingLabels, kernelTraining)
  }

  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  def persist(): Unit = {

    val effectiveTrainingKernel = covariance + noiseModel

    kernelMatrixCache =
      effectiveTrainingKernel.buildKernelMatrix(dataAsIndexSeq(g), npoints)
        .getKernelMatrix()
    //noiseCache = noiseModel.buildKernelMatrix(dataAsIndexSeq(g), npoints)
    //  .getKernelMatrix()
    caching = true
  }

  /**
    * Forget the cached kernel & noise matrices.
    * */
  def unpersist(): Unit = {
    kernelMatrixCache = null
    //noiseCache = null
    caching = false
  }


}

object AbstractSTPRegressionModel {

  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix of the training features
    *
    * */
  def logLikelihood(mu: Double, trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double]): Double = {


    try {
      val dist = MultivariateStudentsT(mu, DenseVector.zeros[Double](trainingData.length), kernelMatrix)
      -1.0*dist.logPdf(trainingData)
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }
}