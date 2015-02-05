package org.kuleuven.esat.optimization

import breeze.linalg.{norm, DenseVector}
import com.tinkerpop.gremlin.scala.{ScalaEdge, ScalaVertex, ScalaGraph}
import org.apache.log4j.{Priority, Logger}

/**
 * Do SGD on a Markov chain to
 * calculate the stationary distribution.
 */
class MarkovChainSGD extends Optimizer[ScalaGraph] {

  private var stepSize: Double = 0.001
  private var numIterations: Int = 100
  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def Mx(g: ScalaGraph)(v: DenseVector[Double]): DenseVector[Double] = {
    DenseVector.tabulate[Double](v.length)((i) => {
      val node = ScalaVertex.wrap(g.getVertex(("state", i)))
      val it = node.getOutEdges("transition").iterator()
      var sum: Double = 0
      while(it.hasNext) {
        val edge = ScalaEdge.wrap(it.next)
        val other_node = edge.getProperty("vertices").asInstanceOf[(Int, Int)]._2
        val prob = edge.getProperty("probability").asInstanceOf[Double]
        sum += prob * v(other_node)
      }
      sum
    })
  }

  override def optimize(g: ScalaGraph, nStates: Int) =
    MarkovChainSGD.runSGD(this.Mx(g), nStates, this.numIterations, this.stepSize)._1


}


object MarkovChainSGD {

  private val logger = Logger.getLogger(this.getClass)

  def runSGD(
      mult: DenseVector[Double] => DenseVector[Double],
      nStates: Int,
      numIterations: Int,
      stepSize: Double): (DenseVector[Double], Double) = {
    val pi = DenseVector.tabulate[Double](nStates)((i) => {if(i == 0) 1.0 else 0.0})
    pi.update(0, 1.0)
    var gradient = DenseVector.zeros[Double](nStates)
    logger.log(Priority.INFO, "Starting SGD on Markov Chain")
    for(i <- 1 to numIterations) {
      gradient = mult(pi) - pi
      pi += (gradient :* (stepSize/math.sqrt(i.toDouble)))
    }
    logger.log(Priority.INFO, "Finished SGD, final residual is: \n"+
      gradient+"\nand result is: "+pi)
    (pi, norm(gradient))
  }
}
