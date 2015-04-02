package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector
import com.tinkerpop.blueprints.{Edge}
import org.apache.log4j.{Logger, Priority}

/**
 * Implements Gradient Descent on the graph
 * generated to calculate approximate optimal
 * values of the model parameters.
 */
class GradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer[Int, DenseVector[Double], DenseVector[Double], Double]{
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 1.0
  private var miniBatchFraction: Double = 1.0

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

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
   * @param xy A function which takes an edge and returns a
   *           [[Tuple2]] of the form (x, y), x being the
   *           predictor vector and y being the target.
   *
   * @return The value of the parameters as a [[DenseVector]]
   *
   *
   * */
  override def optimize(
      nPoints: Int,
      initialP: DenseVector[Double],
      ParamOutEdges: java.lang.Iterable[Edge],
      xy: (Edge) => (DenseVector[Double], Double)): DenseVector[Double] =
    if(this.miniBatchFraction == 1.0) {
      GradientDescent.runSGD(
        nPoints,
        this.regParam,
        this.numIterations,
        this.updater,
        this.gradient,
        this.stepSize,
        initialP,
        ParamOutEdges,
        xy
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
        xy,
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
      POutEdges: java.lang.Iterable[Edge],
      xy: (Edge) => (DenseVector[Double], Double)): DenseVector[Double] = {
    var count = 1
    var oldW: DenseVector[Double] = initial
    var newW = oldW
    val cumGradient: DenseVector[Double] = DenseVector.zeros(initial.length)
    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val targets = POutEdges.iterator()
      while (targets.hasNext) {
        val (x, y) = xy(targets.next())
        gradient.compute(x, y, oldW, cumGradient)
      }
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
      POutEdges: java.lang.Iterable[Edge],
      xy: (Edge) => (DenseVector[Double], Double),
      miniBatchFraction: Double): DenseVector[Double] = {
    var count = 1
    var oldW: DenseVector[Double] = initial
    var newW = oldW
    val cumGradient: DenseVector[Double] = DenseVector.zeros(initial.length)
    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val targets = POutEdges.iterator()
      while (targets.hasNext) {
        if(scala.util.Random.nextDouble() <= miniBatchFraction) {
          val (x, y) = xy(targets.next())
          gradient.compute(x, y, oldW, cumGradient)
        }
      }
      newW = updater.compute(oldW, cumGradient,
        stepSize, count, regParam)._1
      oldW = newW
      count += 1
    }
    newW
  }

}