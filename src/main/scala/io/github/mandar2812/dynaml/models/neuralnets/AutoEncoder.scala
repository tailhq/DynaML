package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.ModelPipe

/**
  * Created by mandar on 23/2/16.
  */
trait AutoEncoder[Source, Data] extends
ModelPipe[Source, Data, DenseVector[Double],
  DenseVector[Double], FeedForwardNetwork[Data]]{

}
