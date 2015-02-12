package org.kuleuven.esat.graphicalModels

import breeze.linalg.{DenseVector, DenseMatrix}
import com.tinkerpop.blueprints.pgm.impls.tg.TinkerGraphFactory
import com.tinkerpop.gremlin.scala.ScalaGraph
import org.apache.log4j.Logger
import org.kuleuven.esat.optimization.MarkovChainSGD

/**
 * Models a Markov Chain transition matrix
 * by a graph and finds an approximate stationary
 * distribution corresponding to it.
 */
private[graphicalModels] class MarkovChain(
    override protected val g: ScalaGraph,
    override protected val nPoints: Int)
  extends GraphicalModel[ScalaGraph]
  with ParameterizedLearner[ScalaGraph, Int, DenseVector[Double]] {

  override protected var params = DenseVector.tabulate[Double](nPoints)((i) => {if(i == 1) 1.0 else 0.0})

  override protected val optimizer = new MarkovChainSGD

  def stationaryDistribution = this.parameters

  def setMaxIterations(i: Int): this.type = {
    this.optimizer.setNumIterations(i)
    this
  }

  def setLearningRate(alpha: Double): this.type = {
    this.optimizer.setStepSize(alpha)
    this
  }

}

object MarkovChain {
  val logger = Logger.getLogger(this.getClass)
  def apply(tr: DenseMatrix[Double]): MarkovChain = {
    val g = TinkerGraphFactory.createTinkerGraph()

    for(i <- 0 to tr.cols - 1) {
      g.addVertex(("state", i))
    }

    tr.foreachPair[Unit]((p:(Int, Int), pr: Double) => {
      g.addEdge(p,
        g.getVertex(("state", p._1)),
        g.getVertex(("state", p._2)),
        "transition")
        .setProperty("probability", pr)
      g.getEdge(p).setProperty("vertices", p)
    })

    new MarkovChain(ScalaGraph.wrap(g), tr.cols)
  }
}
