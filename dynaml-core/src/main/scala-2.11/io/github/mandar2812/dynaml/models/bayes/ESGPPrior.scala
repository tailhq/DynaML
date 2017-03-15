package io.github.mandar2812.dynaml.models.bayes

import io.github.mandar2812.dynaml.DynaMLPipe.sgpTuning
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.analysis.PartitionedVectorField
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.modelpipe.ESGPPipe4
import io.github.mandar2812.dynaml.models.sgp.ESGPModel
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}
import io.github.mandar2812.dynaml.probability.BlockedMESNRV
import spire.algebra.{Field, InnerProductSpace}
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._

import scala.reflect.ClassTag

/**
  * An Extended Skew Gaussian Process Prior over functions.
  *
  * @author mandar2812 date 13/03/2017.
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
        globalOptConfig("gridStep").toDouble) >
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
  * @author mandar2812 date 21/02/2017.
  *
  * An extended skew gaussian process prior with a
  * linear trend function.
  * */
class LinearTrendESGPrior[I: ClassTag](
  cov: LocalScalarKernel[I],
  n: LocalScalarKernel[I],
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
