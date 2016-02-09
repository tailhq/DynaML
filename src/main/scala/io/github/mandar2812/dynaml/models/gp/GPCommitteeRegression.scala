package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.pipes.GPRegressionPipe

/**
  * Created by mandar on 9/2/16.
  */
abstract class GPCommitteeRegression[T,D[T]](num: Int, data: D[T],
                                             networks: GPRegressionPipe[GPRegression, D[T]]*)
  extends LinearModel[D[T], Int, Int, DenseVector[Double], DenseVector[Double],
    Double, D[T]] {

  override protected val g: D[T] = data

  val baseNetworks: List[GPRegression] =
    networks.toList.map(net => net.run(g))

  val num_points = num

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  override def clearParameters(): Unit =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  override def initParams(): DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {

    params = optimizer.optimize(
      num_points,
      g, initParams())
  }

  override protected var params: DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  featureMap = (pattern) =>
    DenseVector(baseNetworks.map(net =>
      net.predictionWithErrorBars(Seq(pattern), 1).head._2
    ).toArray)

}
