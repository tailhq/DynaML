/*
 * Copyright (c) 2016. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 * Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
 * Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
 * Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
 * Vestibulum commodo. Ut rhoncus gravida arcu.
 */

package io.github.mandar2812.dynaml.models.svm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable

/**
  * Created by mandar on 3/4/16.
  */
abstract class AbstractDualLSSVM[Index](data: Stream[(Index, Double)],
                                        numPoints: Int,
                                        kern: CovarianceFunction[Index, Double,
                                          DenseMatrix[Double]])
  extends LinearModel[Stream[(Index, Double)], DenseVector[Double],
    Index, Double, (DenseMatrix[Double], DenseVector[Double])]
    with GloballyOptimizable {

  override protected val g = data

  val kernel = kern

  override protected var current_state: Map[String, Double] =
    Map("regularization" -> 0.1) ++ kernel.state

  override protected var hyper_parameters: List[String] =
    List("regularization")++kernel.hyper_parameters

  val num_points = numPoints

  /**
    * Initialize the synapse weights
    * to small random values between
    * 0 and 1.
    *
    * */
  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](num_points+1)

  override def clearParameters(): Unit = {
    params = initParams()
  }

  override protected var params: DenseVector[Double] = initParams()

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    params = optimizer.optimize(num_points,
      (kernel.buildKernelMatrix(g.map(_._1), num_points).getKernelMatrix(),
        DenseVector(g.map(_._2).toArray)),
      initParams())
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: Index): Double = {

    val features = DenseVector(g.map(inducingpoint =>
      kernel.evaluate(point, inducingpoint._1)).toArray)

    params(0 until num_points) dot features + params(-1)
  }

  def setState(h: Map[String, Double]) = {
    kernel.setHyperParameters(h)
    current_state += ("regularization" -> h("regularization"))
  }


}
