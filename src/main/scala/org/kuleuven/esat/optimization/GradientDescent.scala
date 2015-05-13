package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector
import org.apache.log4j.{Logger, Priority}
import org.kuleuven.esat.graphUtils.CausalEdge
import scala.pickling._
import json._


/**
 * Implements Gradient Descent on the graph
 * generated to calculate approximate optimal
 * values of the model parameters.
 */
class GradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer[Int, DenseVector[Double],
    DenseVector[Double], Double, Iterable[CausalEdge]]{

  private var regParam: Double = 1.0

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Find the optimum value of the parameters using
   * Gradient Descent.
   *
   * @param nPoints The number of data points
   * @param initialP The initial value of the parameters
   *                 as a [[DenseVector]]
   * @param ParamOutEdges An [[java.lang.Iterable]] object
   *                      having all of the out edges of the
   *                      parameter node
   *
   * @return The value of the parameters as a [[DenseVector]]
   *
   *
   * */
  override def optimize(nPoints: Int, initialP: DenseVector[Double],
                        ParamOutEdges: Iterable[CausalEdge])
  : DenseVector[Double] =
    if(this.miniBatchFraction == 1.0) {
      GradientDescent.runSGD(
        nPoints,
        this.regParam,
        this.numIterations,
        this.updater,
        this.gradient,
        this.stepSize,
        initialP,
        ParamOutEdges
      )
    } else {
      GradientDescent.runBatchSGD(
        nPoints,
        this.regParam,
        this.numIterations,
        this.updater,
        this.gradient,
        this.stepSize,
        initialP,
        ParamOutEdges,
        this.miniBatchFraction
      )
    }

}

object GradientDescent {

  private val logger = Logger.getLogger(this.getClass)

  def runSGD(
      nPoints: Int,
      regParam: Double,
      numIterations: Int,
      updater: Updater,
      gradient: Gradient,
      stepSize: Double,
      initial: DenseVector[Double],
      POutEdges: Iterable[CausalEdge]): DenseVector[Double] = {
    var count = 1
    var oldW: DenseVector[Double] = initial
    var newW = oldW

    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val cumGradient: DenseVector[Double] = DenseVector.zeros(initial.length)
      POutEdges.foreach((ed) => {
        val xarr = ed.getPoint().getFeatureMap()
        val x = DenseVector(xarr)
        val y = ed.getLabel().getValue()
        gradient.compute(x, y, oldW, cumGradient)
      })
      newW = updater.compute(oldW, cumGradient,
        stepSize, count, regParam)._1
      oldW = newW
      count += 1
    }
    newW
  }

  def runBatchSGD(
      nPoints: Int,
      regParam: Double,
      numIterations: Int,
      updater: Updater,
      gradient: Gradient,
      stepSize: Double,
      initial: DenseVector[Double],
      POutEdges: Iterable[CausalEdge],
      miniBatchFraction: Double): DenseVector[Double] = {
    var count = 1
    var oldW: DenseVector[Double] = initial
    var newW = oldW
    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val cumGradient: DenseVector[Double] = DenseVector.zeros(initial.length)
      POutEdges.foreach((ed) => {
        if(scala.util.Random.nextDouble() <= miniBatchFraction) {
          val x = DenseVector(ed.getPoint().getFeatureMap())
          val y = ed.getLabel().getValue()
          gradient.compute(x, y, oldW, cumGradient)
        }
      })
      newW = updater.compute(oldW, cumGradient,
        stepSize, count, regParam)._1
      oldW = newW
      count += 1
    }
    newW
  }

}