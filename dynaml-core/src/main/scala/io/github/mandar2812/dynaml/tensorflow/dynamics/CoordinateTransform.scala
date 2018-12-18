package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes.DataPipe
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Layer


private[dynamics] case class CoordinateTransform[I, J](
  name: String,
  transform: Layer[I, J]) extends
  DataPipe[Layer[J, Output], Layer[I, Output]] {

  override def run(data: Layer[J, Output]): Layer[I, Output] = transform >> data

}
