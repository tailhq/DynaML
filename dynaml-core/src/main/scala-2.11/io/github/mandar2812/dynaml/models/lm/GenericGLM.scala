package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.LinearModel

/**
  * @author mandar2812 date: 25/01/2017.
  *
  * The base class for representing single output
  * Generalised Linear Models.
  *
  * See also [[GeneralizedLinearModel]] and [[SparkGLM]]
  */
abstract class GenericGLM[Data, T](
  data: Data, numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]])
  extends LinearModel[Data, DenseVector[Double], DenseVector[Double], Double, T] {

  featureMap = map

  val h: (Double) => Double = identity

  def prepareData(d: Data): T

  override def predict(point: DenseVector[Double]): Double =
    h(params dot DenseVector(featureMap(point).toArray ++ Array(1.0)))

  override def clearParameters(): Unit = {
    params = initParams()
  }

  /**
    * The training data
    **/
  override protected val g: Data = data

  override def learn(): Unit = {
    params = optimizer.optimize(numPoints,
      prepareData(g), initParams())
  }

}
