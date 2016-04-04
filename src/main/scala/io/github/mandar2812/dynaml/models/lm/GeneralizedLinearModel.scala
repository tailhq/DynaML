/*
 * Copyright (c) 2016. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 * Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
 * Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
 * Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
 * Vestibulum commodo. Ut rhoncus gravida arcu.
 */

package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable

/**
  * Created by mandar on 4/4/16.
  */
abstract class GeneralizedLinearModel[T](data: Stream[(DenseVector[Double], Double)],
                             numPoints: Int,
                             map: (DenseVector[Double]) => DenseVector[Double] =
                             identity[DenseVector[Double]] _)
  extends LinearModel[Stream[(DenseVector[Double], Double)],
    DenseVector[Double], DenseVector[Double], Double, T]
    with GloballyOptimizable {

  override protected val g = data

  val h: (Double) => Double = identity _

  def dimensions = featureMap(data.head._1).length

  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](dimensions + 1)


  override protected var params: DenseVector[Double] = initParams()

  override def clearParameters(): Unit = {
    params = initParams()
  }

  def prepareData: T


  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    params = optimizer.optimize(numPoints,
      prepareData, initParams())
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    h(params dot DenseVector(featureMap(point).toArray ++ Array(1.0)))

  override protected var hyper_parameters: List[String] =
    List("regularization")

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double],
                      options: Map[String, String]): Double = {

    setState(h)
    val designMatrix = DenseMatrix.vertcat[Double](
      g.map(point => featureMap(point._1).toDenseMatrix):_*
    )

    val kernelTraining = designMatrix.t*designMatrix
    val trainingLabels = DenseVector(g.map(_._2).toArray)
    val noiseMat = DenseMatrix.eye[Double](dimensions)*h("regularization")

    AbstractGPRegressionModel.logLikelihood(trainingLabels, kernelTraining, noiseMat)
  }

  override protected var current_state: Map[String, Double] = Map("regularization" -> 0.001)

  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type ={
    this.setRegParam(s("regularization"))
    current_state = Map("regularization" -> s("regularization"))
    this
  }
}

object GeneralizedLinearModel {
  def apply[T](data: Stream[(DenseVector[Double], Double)],
               task: String = "regression",
               map: (DenseVector[Double]) => DenseVector[Double] =
               identity[DenseVector[Double]] _,
               modeltype: String = "") = task match {
    case "regression" => new RegularizedGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
    case "classification" => task match {
      case "probit" => new ProbitGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
      case _ => new LogisticGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
    }
  }
}
