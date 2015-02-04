package org.kuleuven.esat.graphicalModels

import breeze.linalg.{reshape, DenseVector}
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.pgm.impls.tg.TinkerGraphFactory
import com.tinkerpop.gremlin.scala.{ScalaEdge, ScalaVertex, ScalaGraph}
import org.apache.log4j.{Priority, Logger}
import org.kuleuven.esat.optimization.{SquaredL2Updater, LeastSquaresGradient, GradientDescent}


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
  private var params = g.getVertex("w").getProperty("slope").asInstanceOf[DenseVector[Double]]
  private var maxIterations: Int = 100
  private var learningRate: Double = 0.001


  private val optimizer = new GradientDescent(
    new LeastSquaresGradient(),
    new SquaredL2Updater())


  def setMaxIterations(i: Int): GaussianLinearModel = {
    this.optimizer.setNumIterations(i)
    this
  }

  def setLearningRate(alpha: Double): GaussianLinearModel = {
    this.optimizer.setStepSize(alpha)
    this
  }

  override def parameters(): DenseVector[Double] =
    this.params

  override def predict(point: DenseVector[Double]) = {
    val point1 = reshape(point, point.length + 1, 1).toDenseVector
    point1.update(point.length, 1)
    this.params dot point1
  }

  override def learn(): Unit = {
    this.params = optimizer.optimize(this.g, nPoints)
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
