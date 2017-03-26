package io.github.mandar2812.dynaml.modelpipe

import io.github.mandar2812.dynaml.models.Model
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Top level trait for Pipes involving ML models.
  */
trait ModelPipe[-Source, T, Q, R, +M <: Model[T, Q, R]]
  extends DataPipe[Source, M] {

  val preProcess: (Source) => T

  override def run(data: Source): M
}

/**
  * A pipeline which encapsulates a DynaML [[Model]]
  * */
class ModelPredictionPipe[T, P, Q, R, S, M <: Model[T, Q, R]](
  pre: DataPipe[P, Q], m: M, po: DataPipe[R, S])
  extends DataPipe[P, S] {

  val preprocess: DataPipe[P, Q] = pre

  val baseModel: M = m

  val postprocess: DataPipe[R, S] = po

  protected val netFlow: DataPipe[P, S] = preprocess > DataPipe((x: Q) => baseModel.predict(x)) > postprocess

  override def run(data: P) = netFlow(data)
}

object ModelPredictionPipe {
  def apply[T, P, Q, R, S, M <: Model[T, Q, R]](pre: DataPipe[P, Q], m: M, po: DataPipe[R, S]) =
    new ModelPredictionPipe[T, P, Q, R, S, M](pre, m, po)
}