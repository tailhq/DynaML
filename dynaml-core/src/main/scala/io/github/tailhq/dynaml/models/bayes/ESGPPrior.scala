package io.github.tailhq.dynaml.models.bayes

import io.github.tailhq.dynaml.DynaMLPipe.sgpTuning
import io.github.tailhq.dynaml.algebra.PartitionedVector
import io.github.tailhq.dynaml.analysis.PartitionedVectorField
import io.github.tailhq.dynaml.kernels.LocalScalarKernel
import io.github.tailhq.dynaml.modelpipe.ESGPPipe4
import io.github.tailhq.dynaml.models.sgp.ESGPModel
import io.github.tailhq.dynaml.pipes.{DataPipe, Encoder, MetaPipe}
import io.github.tailhq.dynaml.probability.BlockedMESNRV
import spire.algebra.{Field, InnerProductSpace}
import io.github.tailhq.dynaml.algebra.PartitionedMatrixOps._

import scala.reflect.ClassTag

/**
  * An Extended Skew Gaussian Process Prior over functions.
  *
  * @author tailhq date 13/03/2017.
  *
  * @tparam I The index set over which the process is defined
  * @tparam MeanFuncParams The parameters specifying the trend function
  *
  * @param covariance The covariance function of the underlying prior
  * @param noiseCovariance The covariance function of the noise.
  * */
abstract class ESGPPrior[I: ClassTag, MeanFuncParams](
  val covariance: LocalScalarKernel[I],
  val noiseCovariance: LocalScalarKernel[I],
  lambda: Double, tau: Double) extends
  StochasticProcessPrior[
    I, Double, PartitionedVector,
    BlockedMESNRV, BlockedMESNRV,
    ESGPModel[Seq[(I, Double)], I]] {

  type ExtendedSGPModel = ESGPModel[Seq[(I, Double)], I]

  def _meanFuncParams: MeanFuncParams

  def meanFuncParams_(p: MeanFuncParams): Unit

  private val initial_state =
    covariance.state ++ noiseCovariance.state ++ Map("skewness" -> lambda, "cutoff" -> tau)

  val meanFunctionPipe: MetaPipe[MeanFuncParams, I, Double]

  val trendParamsEncoder: Encoder[MeanFuncParams, Map[String, Double]]

  private var globalOptConfig = Map(
    "globalOpt" -> "GS",
    "gridSize" -> "3",
    "gridStep" -> "0.2")

  /**
    * Append the global optimization configuration
    * */
  def globalOptConfig_(conf: Map[String, String]) = globalOptConfig ++= conf


  /**
    * Data pipe which takes as input training data and a trend model,
    * outputs a tuned gaussian process regression model.
    * */
  def posteriorModelPipe =
    ESGPPipe4[I](covariance, noiseCovariance) >
      sgpTuning(
        initial_state,
        globalOptConfig("globalOpt"),
        globalOptConfig("gridSize").toInt,
        globalOptConfig("gridStep").toDouble,
        prior = hyperPrior) >
      DataPipe((modelAndConf: (ExtendedSGPModel, Map[String, Double])) => modelAndConf._1)

  /**
    * Given some data, return a gaussian process regression model
    *
    * @param data A Sequence of input patterns and responses
    * */
  override def posteriorModel(data: Seq[(I, Double)]) =
    posteriorModelPipe(meanFunctionPipe(_meanFuncParams), lambda, tau, data)

  /**
    * Returns the distribution of response values,
    * evaluated over a set of domain points of type [[I]].
    * */
  override def priorDistribution[U <: Seq[I]](d: U) = {

    val numPoints: Long = d.length.toLong

    //Declare vector field, required implicit parameter
    implicit val field: Field[PartitionedVector] =
      PartitionedVectorField(numPoints, covariance.rowBlocking)

    //Construct mean Vector
    val meanFunc = meanFunctionPipe(_meanFuncParams)
    val meanVector = PartitionedVector(
      d.toStream.map(meanFunc(_)),
      numPoints,
      covariance.rowBlocking)

    val effectiveCov = covariance + noiseCovariance
    //Construct covariance matrix
    val covMat = effectiveCov.buildBlockedKernelMatrix(d, numPoints)
    val lVec: PartitionedVector = field.one*lambda

    BlockedMESNRV(tau, lVec, meanVector, covMat)
  }

}

/**
  * An extended skew gaussian process prior with a
  * linear trend function.
  *
  * @author tailhq date 21/02/2017.
  * */
class LinearTrendESGPrior[I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  override val trendParamsEncoder: Encoder[(I, Double), Map[String, Double]],
  lambda: Double, tau: Double,
  trendParams: I, intercept: Double)(
  implicit inner: InnerProductSpace[I, Double]) extends
  ESGPPrior[I, (I, Double)](cov, n, lambda, tau) with
  LinearTrendStochasticPrior[
    I, BlockedMESNRV, BlockedMESNRV,
    ESGPModel[Seq[(I, Double)], I]]{

  override val innerProduct = inner

  override protected var params: (I, Double) = (trendParams, intercept)

  override def _meanFuncParams = params

  override def meanFuncParams_(p: (I, Double)) = params = p

  override val meanFunctionPipe = MetaPipe(
    (parameters: (I, Double)) => (x: I) => inner.dot(parameters._1, x) + parameters._2
  )
}
