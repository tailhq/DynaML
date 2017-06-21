package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.models.StochasticProcessMixtureModel
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GaussianProcessMixture}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability.MultGaussianPRV
import io.github.mandar2812.dynaml.probability.distributions.BlockedMultiVariateGaussian

import scala.reflect.ClassTag

/**
  * Constructs a gaussian process mixture model
  * from a single [[AbstractGPRegressionModel]] instance.
  * @tparam T The type of the GP training data
  * @tparam I The index set/input domain of the GP model.
  * @author mandar2812 date 15/06/2017.
  * */
class ProbGPMixtureMachine[T, I: ClassTag](
  model: AbstractGPRegressionModel[T, I]) extends
  MixtureMachine[T, I, Double, PartitionedVector, PartitionedPSDMatrix, BlockedMultiVariateGaussian,
    MultGaussianPRV, AbstractGPRegressionModel[T, I]](model) {


  val (kernelPipe, noisePipe) = (system.covariance.asPipe, system.noiseModel.asPipe)

  def blockedHypParams = system.covariance.blocked_hyper_parameters ++ system.noiseModel.blocked_hyper_parameters

  def blockedState = system._current_state.filterKeys(blockedHypParams.contains)

  implicit val transform: DataPipe[T, Seq[(I, Double)]] = DataPipe(system.dataAsSeq _)

  override val confToModel = DataPipe((model_state: Map[String, Double]) =>
    AbstractGPRegressionModel(
    kernelPipe(model_state), noisePipe(model_state),
    system.mean)(system.data, system.npoints))

  override val mixturePipe = DataPipe2(
    (models: Seq[AbstractGPRegressionModel[T, I]], weights: DenseVector[Double]) =>
    StochasticProcessMixtureModel[T, I](models, weights))
  
}
