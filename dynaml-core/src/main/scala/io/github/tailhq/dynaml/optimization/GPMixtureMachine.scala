package io.github.tailhq.dynaml.optimization

import io.github.tailhq.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.tailhq.dynaml.modelpipe._
import io.github.tailhq.dynaml.models.gp.AbstractGPRegressionModel
import io.github.tailhq.dynaml.pipes.DataPipe
import io.github.tailhq.dynaml.probability.MultGaussianPRV
import io.github.tailhq.dynaml.probability.distributions.BlockedMultiVariateGaussian

import scala.reflect.ClassTag

/**
  * Constructs a gaussian process mixture model
  * from a single [[AbstractGPRegressionModel]] instance.
  * @tparam T The type of the GP training data
  * @tparam I The index set/input domain of the GP model.
  * @author tailhq date 15/06/2017.
  * */
class GPMixtureMachine[T, I: ClassTag](
  model: AbstractGPRegressionModel[T, I]) extends
  MixtureMachine[
    T, I, Double, PartitionedVector, PartitionedPSDMatrix,
    BlockedMultiVariateGaussian, MultGaussianPRV,
    AbstractGPRegressionModel[T, I]](model) {

  val (kernelPipe, noisePipe) = (system.covariance.asPipe, system.noiseModel.asPipe)

  def blockedHypParams = system.covariance.blocked_hyper_parameters ++ system.noiseModel.blocked_hyper_parameters

  def blockedState = system._current_state.filterKeys(blockedHypParams.contains)

  implicit val transform: DataPipe[T, Seq[(I, Double)]] = DataPipe(system.dataAsSeq)

  override val confToModel = DataPipe(
    (model_state: Map[String, Double]) =>
      AbstractGPRegressionModel(
        kernelPipe(model_state), noisePipe(model_state),
        system.mean)(system.data, system.npoints)
  )

  override val mixturePipe = new GPMixturePipe[T, I]

}
