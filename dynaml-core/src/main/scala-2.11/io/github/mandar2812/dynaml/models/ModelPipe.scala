package io.github.mandar2812.dynaml.models

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Top level trait for Pipes involving ML models.
  */
trait ModelPipe[Source, T, Q, R, M <: Model[T, Q, R]]
  extends DataPipe[Source, M] {

  val preProcess: (Source) => T

  override def run(data: Source): M
}
