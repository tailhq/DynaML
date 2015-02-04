package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector
import com.tinkerpop.gremlin.scala.{ScalaEdge, ScalaVertex, ScalaGraph}
import org.apache.log4j.{Logger, Priority}

/**
 * Implements Gradient Descent on the graph
 * generated to calculate approximate optimal
 * values of the model parameters.
 */
class GradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer[ScalaGraph]{
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

  override def optimize(g: ScalaGraph, nPoints: Int): DenseVector[Double] =
    if(this.miniBatchFraction == 1.0) {
      GradientDescent.runSGD(
        g,
        nPoints,
        this.regParam,
        this.numIterations,
        this.updater,
        this.gradient,
        this.stepSize
      )
    } else {
      GradientDescent.runBatchSGD(
        g,
        nPoints,
        this.regParam,
        this.numIterations,
        this.updater,
        this.gradient,
        this.stepSize,
        this.miniBatchFraction
      )
    }

}

object GradientDescent {
  private val logger = Logger.getLogger(this.getClass)
  def runSGD(
      g: ScalaGraph,
      nPoints: Int,
      regParam: Double,
      numIterations: Int,
      updater: Updater,
      gradient: Gradient,
      stepSize: Double): DenseVector[Double] = {
    var count = 1
    val w = ScalaVertex.wrap(g.getVertex("w"))
    var oldW: DenseVector[Double] = w.getProperty("slope").asInstanceOf[DenseVector[Double]]
    var newW: DenseVector[Double] = oldW
    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val targets = w.getOutEdges().iterator()
      while (targets.hasNext) {
        val edge = ScalaEdge.wrap(targets.next())
        val yV = ScalaVertex.wrap(edge.getInVertex)
        val y = yV.getProperty("value").asInstanceOf[Double]

        val xV = yV.getInEdges("causes").iterator().next().getOutVertex
        val x = xV.getProperty("value").asInstanceOf[DenseVector[Double]]

        val (grad, _): (DenseVector[Double], Double) = gradient.compute(x, y, oldW)

        newW = updater.compute(oldW, grad, stepSize, count, regParam)._1
        oldW = newW
      }
      count += 1
    }

    w.setProperty("slope", newW)
    newW
  }

  def runBatchSGD(
      g: ScalaGraph,
      nPoints: Int,
      regParam: Double,
      numIterations: Int,
      updater: Updater,
      gradient: Gradient,
      stepSize: Double,
      miniBatchFraction: Double): DenseVector[Double] = {
    DenseVector.zeros[Double](10)
  }

}