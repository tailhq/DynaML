package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes.DataPipe
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.layers.Layer


private[dynamics] case class CoordinateTransform[I, J, D: TF: IsNotQuantized](
  name: String,
  transform: Layer[I, J]) extends
  DataPipe[Layer[J, Output[D]], Layer[I, Output[D]]] {

  override def run(data: Layer[J, Output[D]]): Layer[I, Output[D]] = transform >> data

}
