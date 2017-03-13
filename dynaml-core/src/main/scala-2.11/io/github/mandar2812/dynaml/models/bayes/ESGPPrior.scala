package io.github.mandar2812.dynaml.models.bayes

import io.github.mandar2812.dynaml.DynaMLPipe.sgpTuning
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.analysis.PartitionedVectorField
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.modelpipe.ESGPPipe4
import io.github.mandar2812.dynaml.models.sgp.ESGPModel
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}
import io.github.mandar2812.dynaml.probability.BlockedMESNRV
import spire.algebra.Field
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import scala.reflect.ClassTag

/**
  * Created by mandar on 13/03/2017.
  */
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
