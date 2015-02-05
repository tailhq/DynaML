package org.kuleuven.esat.graphicalModels

import breeze.linalg.DenseVector
import org.kuleuven.esat.optimization.Optimizer


/**
 * Basic Higher Level abstraction
 * for graphical models.
 *
 */
trait GraphicalModel[T] {
  protected val g: T
}

trait ParameterizedLearner[G] extends GraphicalModel[G] {
  protected var params: DenseVector[Double]
  protected val optimizer: Optimizer[G]
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

trait LinearModel[T, P]
  extends GraphicalModel[T]
  with ParameterizedLearner[T] {

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: P): Double


}
