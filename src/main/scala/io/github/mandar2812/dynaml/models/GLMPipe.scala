package io.github.mandar2812.dynaml.models

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel

/**
  * Created by mandar on 15/6/16.
  */
class GLMPipe[T, Source](pre: (Source) => Stream[(DenseVector[Double], Double)],
                         map: (DenseVector[Double]) => (DenseVector[Double]) = identity _,
                         task: String = "regression", modelType: String = "") extends
  ModelPipe[Source, Stream[(DenseVector[Double], Double)],
    DenseVector[Double], Double,
    GeneralizedLinearModel[T]] {

  override val preProcess = pre

  override def run(data: Source) = {
    val training = preProcess(data)
    GeneralizedLinearModel[T](training, task, map, modelType)
  }

}
