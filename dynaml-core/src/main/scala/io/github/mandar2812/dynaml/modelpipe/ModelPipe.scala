package io.github.mandar2812.dynaml.modelpipe

import io.github.mandar2812.dynaml.models.Model
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes.{DataPipe, ReversibleScaler}

/**
  * Top level trait for Pipes returning ML models.
  */
trait ModelPipe[-Source, T, Q, R, +M <: Model[T, Q, R]]
  extends DataPipe[Source, M] {

  val preProcess: (Source) => T

  override def run(data: Source): M
}

/**
  * A pipeline which encapsulates a DynaML [[Model.predict()]] functionality.
  *
  * @tparam T The training data type accepted by the encapsulated model
  * @tparam P The type of unprocessed input to the pipe
  * @tparam Q The type of input features the model accepts
  * @tparam R The type of output returned by [[Model.predict()]]
  * @tparam S The type of the processed output.
  *
  * @param m The underlying model
  * @param pre Pre-processing [[DataPipe]]
  * @param po Post-processing [[DataPipe]]
  *
  * */
class ModelPredictionPipe[T, -P, Q, R, +S, M <: Model[T, Q, R]](
  pre: DataPipe[P, Q], m: M, po: DataPipe[R, S])
  extends DataPipe[P, S] {

  val preprocess: DataPipe[P, Q] = pre

  val baseModel: M = m

  val postprocess: DataPipe[R, S] = po

  protected val netFlow: DataPipe[P, S] =
    preprocess > DataPipe((x: Q) => baseModel.predict(x)) > postprocess

  override def run(data: P) = netFlow(data)
}

object ModelPredictionPipe {
  /**
    * Create a [[ModelPredictionPipe]] instance given
    * a pre-processing flow, a DynaML [[Model]] and a post-processing flow
    * respectively.
    * */
  def apply[T, P, Q, R, S, M <: Model[T, Q, R]](
    pre: DataPipe[P, Q], m: M, po: DataPipe[R, S]) =
    new ModelPredictionPipe[T, P, Q, R, S, M](pre, m, po)

  /**
    * Create a [[ModelPredictionPipe]] instance
    * (having no pre or post processing steps)
    * given a DynaML [[Model]]
    *
    * */
  def apply[T, Q, R, M <: Model[T, Q, R]](m: M) =
    new ModelPredictionPipe[T, Q, Q, R, R, M](identityPipe[Q], m, identityPipe[R])

  /**
    * Create a [[ModelPredictionPipe]] instance
    * given scaling relationships for features and outputs,
    * along with a DynaML [[Model]]
    *
    * */
  def apply[T, Q, R, M <: Model[T, Q, R]](
    featuresSc: ReversibleScaler[Q], outputSc: ReversibleScaler[R], m: M) =
    new ModelPredictionPipe[T, Q, Q, R, R, M](featuresSc, m, outputSc.i)

}
