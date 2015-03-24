package org.kuleuven.esat.graphicalModels

import breeze.linalg.{DenseVector, DenseMatrix}
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.{Graph, Direction, Edge}
import com.tinkerpop.blueprints.impls.tg.{TinkerGraphFactory}
import com.tinkerpop.gremlin.scala.{ScalaGraph, ScalaEdge, ScalaVertex}
import org.apache.log4j.{Priority, Logger}
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.kernels.{GaussianDensityKernel, SVMKernel}
import org.kuleuven.esat.optimization._
import org.kuleuven.esat.prototype.{QuadraticRenyiEntropy, GreedyEntropySelector}
import org.kuleuven.esat.utils


/**
 * Linear Model with conditional probability
 * of the target variable given the features
 * is a Gaussian with mean = wT.x.
 *
 * Gaussian priors on the parameters are imposed
 * as a means for L2 regularization.
 */

private[esat] class GaussianLinearModel(
    override protected val g: Graph,
    override protected val nPoints: Int,
    private val featuredims: Int,
    implicit val task: String)
  extends LinearModel[Graph, Int, Int,
    DenseVector[Double], DenseVector[Double], Double] {

  private val logger = Logger.getLogger(this.getClass)

  override implicit protected var params =
    g.getVertex("w").getProperty("slope").asInstanceOf[DenseVector[Double]]

  override protected val optimizer = GaussianLinearModel.getOptimizer(task)

  def npoints = nPoints

  def setMaxIterations(i: Int): this.type = {
    this.optimizer.setNumIterations(i)
    this
  }

  def setLearningRate(alpha: Double): this.type = {
    this.optimizer.setStepSize(alpha)
    this
  }

  def setBatchFraction(f: Double): this.type = {
    assert(f >= 0.0 && f <= 1.0, "Mini-Batch Fraction should be between 0.0 and 1.0")
    this.optimizer.setMiniBatchFraction(f)
    this
  }

  def setRegParam(reg: Double): this.type = {
    this.optimizer.setRegParam(reg)
    this
  }

  override def parameters(): DenseVector[Double] =
    this.params

  def score(point: DenseVector[Double]): Double =
    GaussianLinearModel.score(this.params)(this.featureMap(List(point))(0))

  override def predict(point: DenseVector[Double]): Double = task match {
    case "classification" => math.signum(this.score(point))
    case "regression" => this.score(point)
  }

  override def getParamOutEdges() = this.g.getVertex("w").getEdges(Direction.OUT)

  override def getxyPair(ed: Edge): (DenseVector[Double], Double) = {
    val edge = ScalaEdge.wrap(ed)
    val yV = ScalaVertex.wrap(edge.getVertex(Direction.IN))
    val y = yV.getProperty("value").asInstanceOf[Double]

    val xV = yV.getEdges(Direction.IN, "causes").iterator().next().getVertex(Direction.OUT)
    val x = xV.getProperty("featureMap").asInstanceOf[DenseVector[Double]]
    (x, y)
  }

  override def applyKernel(kernel: SVMKernel[DenseMatrix[Double]],
                           M: Int = npoints): Unit = {
    //Get the original features of the data
    val features = this.filter((_) => true)
    var points = features
    if(M < this.npoints) {
      logger.log(Priority.INFO, "Calculating sample variance of the data set")

      //Calculate the column means and variances
      val (mean, variance) = utils.getStats(features)

      //Use the adjusted value of the variance
      val adjvarance:DenseVector[Double] = variance :/= npoints.toDouble
      val density = new GaussianDensityKernel

      logger.log(Priority.INFO,
        "Using Silvermans rule of thumb to set bandwidth of density kernel")
      density.setBandwidth(DenseVector.tabulate[Double](featuredims - 1){
        i => 1.06*math.sqrt(adjvarance(i))/math.pow(npoints, 0.2)
      })
      val subsetsel = new GreedyEntropySelector(new QuadraticRenyiEntropy(density))

      logger.log(Priority.INFO,
        "Building low rank appriximation to kernel matrix")
      points = subsetsel.selectPrototypes(this, M)
    } else {
      points = features
    }

    val kernelMatrix =
      kernel.buildKernelMatrix(points, M)
    val decomposition = kernelMatrix.eigenDecomposition(features.length)
    this.featureMap = kernel.featureMapping(decomposition)(features) _
    val edges = this.getParamOutEdges().iterator()
    this.params = DenseVector.ones[Double](decomposition._1.length + 1)
    this.g.getVertex("w").setProperty("slope", this.params)
    while (edges.hasNext) {
      //Get the predictor vertex corresponding to the edge
      val vertex = edges.next().getVertex(Direction.IN)
        .getEdges(Direction.IN, "causes").iterator()
        .next().getVertex(Direction.OUT)

      //Get the original features of the point
      val featurex = vertex.getProperty("value").asInstanceOf[DenseVector[Double]]

      //Get mapped features for the point
      val mappedf = featureMap(List(featurex(0 to featurex.length - 2)))(0)
      val newFeatures = DenseVector.vertcat[Double](mappedf, DenseVector(Array(1.0)))

      //Set a new property in the vertex corresponding to the mapped features
      vertex.setProperty("featureMap", newFeatures)
    }
  }

  override def evaluate(reader: CSVReader, head: Boolean): Metrics[Double] =
    GaussianLinearModel.evaluate(this.featureMap)(this.params)(reader, head)

  /**
   * Override the effect of appyling a kernel
   * and return the model back to its default
   * state i.e. the Identity Kernel
   * */
  override def clearParameters(): Unit = {
    this.params = DenseVector.ones[Double](this.featuredims)
    this.featureMap = (x) => x
    val it = this.getParamOutEdges().iterator()
    while(it.hasNext) {
      val outEdge = it.next()
      val ynode = outEdge.getVertex(Direction.IN)
      val xnode = ynode.getEdges(Direction.IN,"causes")
        .iterator().next().getVertex(Direction.OUT)
      xnode.setProperty(
        "featureMap",
        xnode.getProperty("value")
          .asInstanceOf[DenseVector[Double]]
      )
      this.g.getVertex("w").setProperty("slope", this.params)
    }
  }

  def filter(fn : (Int) => Boolean): List[DenseVector[Double]] =
    (1 to npoints).filter(fn).map{
      i =>
        this.g.getVertex(("x", i)).getProperty("value")
          .asInstanceOf[DenseVector[Double]](0 to featuredims - 2)
    }.toList

}

object GaussianLinearModel {

  val logger = Logger.getLogger(this.getClass)

  /**
   * Factory method to create the appropriate
   * optimization object required for the Gaussian
   * model
   * */
  def getOptimizer(task: String): GradientDescent = task match {
    case "classification" => new GradientDescent(
      new LeastSquaresSVMGradient(),
      new SquaredL2Updater())

    case "regression" => new GradientDescent(
      new LeastSquaresGradient(),
      new SquaredL2Updater())
  }

  /**
   * A curried function which calculates the predicted
   * value of the target variable using the parameters
   * and the point in question.
   *
   * This is used for the implementation of the method
   * score in [[GaussianLinearModel]]
   * */

  def score(params: DenseVector[Double])
           (point: DenseVector[Double]): Double =
    params(0 to params.length-2) dot point +
    params(params.length-1)

  /**
   * The actual implementation of the evaluate
   * feature.
   * */
  def evaluate(featureMap: (List[DenseVector[Double]]) => List[DenseVector[Double]])
              (params: DenseVector[Double])
              (reader: CSVReader, head: Boolean)
              (implicit task: String): Metrics[Double] = {
    val lines = reader.iterator
    var index = 1
    var dim = 0
    if(head) {
      dim = lines.next().length
    }

    logger.log(Priority.INFO, "Calculating test set predictions")

    val scoresAndLabels = lines.map{line =>
      //Parse line and extract features

      if (dim == 0) {
        dim = line.length
      }

      val yv = line.apply(line.length - 1).toDouble
      val xv: DenseVector[Double] =
        DenseVector(line.slice(0, line.length - 1).toList.map{x => x.toDouble}.toArray)
      (score(params)(featureMap(List(xv))(0)), yv)

    }.toList

    Metrics(task)(scoresAndLabels)
  }

  def apply(reader: CSVReader, head: Boolean, task: String): GaussianLinearModel = {
    val g = TinkerGraphFactory.createTinkerGraph()
    val lines = reader.iterator
    var index = 1
    var dim = 0
    if(head) {
      dim = lines.next().length
    }

    logger.log(Priority.INFO, "Creating graph for data set.")
    g.addVertex("w").setProperty("variable", "parameter")

    while (lines.hasNext) {
      //Parse line and extract features
      val line = lines.next()
      if(dim == 0) {
        dim = line.length
      }

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
      g.getVertex(("x", index)).setProperty("featureMap", xv)
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

    g.getVertex("w").setProperty("slope", DenseVector.ones[Double](dim))

    logger.log(Priority.INFO, "Graph constructed, now building model object.")
    new GaussianLinearModel(g, index-1, dim, task)
  }
}
