package io.github.mandar2812.dynaml.models

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
trait ModelPredictionPipe[T, Q, R, M <: Model[T, Q, R]]
  extends DataPipe[Q, R] {

  val baseModel: M

  override def run(data: Q) = baseModel.predict(data)
}

object ModelPredictionPipe {
  def apply[T, Q, R, M <: Model[T, Q, R]](model: M) = new ModelPredictionPipe[T, Q, R, M] {
    override val baseModel = model
  }
}