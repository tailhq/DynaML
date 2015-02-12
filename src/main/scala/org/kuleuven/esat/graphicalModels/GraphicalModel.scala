package org.kuleuven.esat.graphicalModels

import breeze.generic.UFunc
import breeze.linalg.operators.OpMulInner
import breeze.linalg.{Tensor, VectorLike, Matrix, DenseVector}
import org.kuleuven.esat.optimization.Optimizer


/**
 * Basic Higher Level abstraction
 * for graphical models.
 *
 */
trait GraphicalModel[T] {
  protected val g: T
}

trait ParameterizedLearner[G, K, T <: Tensor[K, Double]]
  extends GraphicalModel[G] {
  protected var params: T
  protected val optimizer: Optimizer[G, K, T]
  protected val nPoints: Int

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   * */
  def learn(): Unit = {
    this.params = optimizer.optimize(this.g, nPoints)
  }

  /**
   * Get the value of the parameters
   * of the model.
   * */
  def parameters() = this.params

}

/**
 * Represents skeleton of a
 * Linear Model.
 *
 * */

abstract class LinearModel[T, K1, K2,
  P <: Tensor[K1, Double], Q <: Tensor[K2, Double], R]
  extends GraphicalModel[T]
  with ParameterizedLearner[T, K1, P] {

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: Q): R

}
