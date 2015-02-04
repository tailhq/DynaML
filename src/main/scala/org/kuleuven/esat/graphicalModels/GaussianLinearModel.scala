package org.kuleuven.esat.graphicalModels

import breeze.linalg.{reshape, DenseVector}
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.pgm.impls.tg.TinkerGraphFactory
import com.tinkerpop.gremlin.scala.{ScalaEdge, ScalaVertex, ScalaGraph}
import org.apache.log4j.{Priority, Logger}


/**
 * Linear Model with conditional probability
 * of the target variable given the features
 * is a Gaussian with mean = wT.x.
 *
 * Gaussian priors on the parameters are imposed
 * as a means for L2 regularization.
 */

class GaussianLinearModel(
    override protected val g: ScalaGraph,
    private val nPoints: Int)
  extends LinearModel[ScalaGraph, DenseVector[Double]] {

  private val logger = Logger.getLogger(this.getClass)
  private var maxIterations: Int = 100
  private var learningRate: Double = 0.001

  def setMaxIterations(i: Int): GaussianLinearModel = {
    this.maxIterations = i
    this
  }

  def setLearningRate(alpha: Double): GaussianLinearModel = {
    this.learningRate = alpha
    this
  }

  override def parameters(): DenseVector[Double] =
    g.getVertex("w").getProperty("slope").asInstanceOf[DenseVector[Double]]

  override def predict(point: DenseVector[Double]) = {
    val point1 = reshape(point, point.length + 1, 1).toDenseVector
    point1.update(point.length, 1)
    this.parameters dot point1
  }

  override def learn(): Unit = {
    var count = 1
    var w = ScalaVertex.wrap(g.getVertex("w"))
    var oldW: DenseVector[Double] = w.getProperty("slope").asInstanceOf[DenseVector[Double]]
    var newW: DenseVector[Double] = DenseVector.zeros(4)
    var diff: DenseVector[Double] = DenseVector.zeros(4)
    logger.log(Priority.INFO, "Training model using SGD")

    while(count <= this.maxIterations) {
      val targets = w.getOutEdges().iterator()
      while (targets.hasNext) {
        w = g.getVertex("w")
        oldW = w.getProperty("slope").asInstanceOf[DenseVector[Double]]
        val edge = ScalaEdge.wrap(targets.next())
        val yV = ScalaVertex.wrap(edge.getInVertex)
        val y = yV.getProperty("value").asInstanceOf[Double]

        val xV = yV.getInEdges("causes").iterator().next().getOutVertex
        val x = xV.getProperty("value").asInstanceOf[DenseVector[Double]]
        val dt: Double = oldW dot x
        val grad: Double = (dt - y)/nPoints.toDouble

        diff = DenseVector.tabulate(oldW.length)((i) =>
          2*this.learningRate*((x(i) * grad) + oldW(i)))
        newW = oldW - diff
        g.getVertex("w").setProperty("slope", newW)
      }
      count += 1
    }
  }
}

object GaussianLinearModel {

  val logger = Logger.getLogger(this.getClass)

  def apply(reader: CSVReader): GaussianLinearModel = {
    val g = TinkerGraphFactory.createTinkerGraph()
    val head: Boolean = true
    val lines = reader.iterator
    var index = 1
    var dim = 0
    if(head) {
      dim = lines.next().length
    }

    logger.log(Priority.INFO, "Creating graph for data set.")
    g.addVertex("w").setProperty("variable", "parameter")
    g.getVertex("w").setProperty("slope", DenseVector.ones[Double](dim))


    while (lines.hasNext) {
      //Parse line and extract features
      val line = lines.next()
      val yv = line.apply(line.length - 1).toDouble
      val features = line.map((s) => s.toDouble).toArray
      features.update(line.length - 1, 1.0)
      val xv: DenseVector[Double] =
        new DenseVector[Double](features)

      /*
      * Create nodes xi and yi
      * append to them their values
      * properties, etc
      * */
      g.addVertex(("x", index)).setProperty("value", xv)
      g.getVertex(("x", index)).setProperty("variable", "data")

      g.addVertex(("y", index)).setProperty("value", yv)
      g.getVertex(("y", index)).setProperty("variable", "target")

      //Add edge between xi and yi
      g.addEdge((("x", index), ("y", index)),
        g.getVertex(("x", index)), g.getVertex(("y", index)),
        "causes")

      //Add edge between w and y_i
      g.addEdge(("w", ("y", index)), g.getVertex("w"),
        g.getVertex(("y", index)),
        "controls")

      index += 1
    }
    logger.log(Priority.INFO, "Graph constructed, now building model object.")
    new GaussianLinearModel(ScalaGraph.wrap(g), index)
  }
}
