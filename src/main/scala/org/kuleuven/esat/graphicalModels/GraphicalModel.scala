package org.kuleuven.esat.graphicalModels


/**
 * Basic Higher Level abstraction
 * for graphical models.
 *
 */
trait GraphicalModel[T] {
  protected val g: T
}

/**
 * Represents skeleton of a
 * Linear Model.
 *
 * */

trait LinearModel[T, P] extends GraphicalModel[T] {
  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   * */
  def learn(): Unit

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: P): Double

  /**
   * Get the value of the parameters
   * of the model.
   * */
  def parameters(): P
}
