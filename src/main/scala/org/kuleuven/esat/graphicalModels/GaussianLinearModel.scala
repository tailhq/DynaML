package org.kuleuven.esat.graphicalModels

import breeze.linalg.{cholesky, inv, DenseMatrix, DenseVector}
import breeze.numerics.{sigmoid, sqrt}
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.util.io.graphml.GraphMLWriter
import com.tinkerpop.blueprints.util.io.graphson.GraphSONWriter
import com.tinkerpop.blueprints.{GraphFactory, Graph}
import com.tinkerpop.frames.{FramedGraph, FramedGraphFactory}
import com.typesafe.config.ConfigFactory
import org.apache.log4j.{Priority, Logger}
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.optimization._
import org.kuleuven.esat.utils
import scala.collection.mutable
import scala.pickling._
import json._
import collection.JavaConversions._
import org.kuleuven.esat.graphUtils._

/**
 * Linear Model with conditional probability
 * of the target variable given the features
 * is a Gaussian with mean = wT.x.
 *
 * Gaussian priors on the parameters are imposed
 * as a means for L2 regularization.
 */

class GaussianLinearModel(
    override protected val g: FramedGraph[Graph],
    override protected val nPoints: Int,
    override protected val featuredims: Int,
    override protected val vertexMaps: (mutable.HashMap[String, AnyRef],
        mutable.HashMap[Int, AnyRef],
        mutable.HashMap[Int, AnyRef]),
    override protected val edgeMaps: (mutable.HashMap[Int, AnyRef],
      mutable.HashMap[Int, AnyRef]),
    override implicit protected val task: String)
  extends KernelBayesianModel {

  override protected val logger = Logger.getLogger(this.getClass)

  override implicit protected var params =
    DenseVector.ones[Double](featuredims)

  private val (mean, variance) = utils.getStatsMult(this.filter(_ => true))

  override protected val optimizer = GaussianLinearModel.getOptimizer(task)

  private val sigmaInverse: DenseMatrix[Double] = inv(cholesky(variance))

  private val rescale = GaussianLinearModel.scaleAttributes(this.mean, sigmaInverse) _

  def score(point: DenseVector[Double]): Double =
    GaussianLinearModel.score(this.params)(this.featureMap(List(rescale(point))).head)

  override def predict(point: DenseVector[Double]): Double = task match {
    case "classification" => math.tanh(this.score(point))
    case "regression" => this.score(point)
  }

  override def evaluate(config: Map[String, String]): Metrics[Double] = {
    val (file, delim, head, _) = GaussianLinearModel.readConfig(config)
    GaussianLinearModel.evaluate(this.featureMap)(this.params)(
      utils.getCSVReader(file, delim), head
    )
  }

  override def filter(fn : (Int) => Boolean): List[DenseVector[Double]] =
    super.filter(fn).map((p) => p(0 to featuredims - 2))

  /**
   * Saves the underlying graph object
   * in the given file format Supported
   * formats include.
   *
   * 1) Graph JSON: "json"
   * 2) GraphML: "xml"
   * */
  def save(file: String, format: String = "json"): Unit = format match {
    case "json" => GraphSONWriter.outputGraph(this.g, file)
    case "xml" => GraphMLWriter.outputGraph(this.g, file)
  }

  def normalizeData: this.type = {
    logger.info("Rescaling data attributes")
    val xMap = this.vertexMaps._2
    (1 to this.nPoints).foreach{i =>
      val xVertex: Point = this.g.getVertex(xMap(i), classOf[Point])
      val vec: DenseVector[Double] =
        rescale(DenseVector(xVertex.getValue())(0 to featuredims - 2))
      xVertex.setValue(DenseVector.vertcat(vec, DenseVector(1.0)).toArray)
    }
    this
  }
}

object GaussianLinearModel {
  val manager: FramedGraphFactory = new FramedGraphFactory
  val conf = ConfigFactory.load("conf/bayesLearn.conf")
  val logger = Logger.getLogger(this.getClass)

  /**
   * Factory function to rescale attributes
   * given a vector of means and the Cholesky
   * factorization of the inverse variance matrix
   *
   * */
  def scaleAttributes(mean: DenseVector[Double],
                      sigmaInverse: DenseMatrix[Double])(x: DenseVector[Double])
  : DenseVector[Double] = sigmaInverse * (x - mean)

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
    logger.log(Priority.INFO, "Calculating test set predictions")
    val (points, dim) = readCSV(reader, head)
    var index = 1
    val scoresAndLabels = points.map{couple =>
      val yv = couple._2
      val xv = couple._1
      index += 1
      (score(params)(featureMap(List(xv)).head), yv)
    }.toList

    Metrics(task)(scoresAndLabels, index)
  }

  def readCSV(reader: CSVReader, head: Boolean):
  (Iterable[(DenseVector[Double], Double)], Int) = {
    val stream = reader.toStream().toIterable
    val dim = stream.head.length

    def lines = if(head) {
      stream.drop(1)
    } else {
      stream
    }

    (lines.map{parseLine}, dim)
  }

  def parseLine = {line : List[String] =>
    //Parse line and extract features
    val yv = line.apply(line.length - 1).toDouble
    val xv: DenseVector[Double] =
      DenseVector(line.slice(0, line.length - 1).map{x => x.toDouble}.toArray)

    (xv, yv)
  }

  def readConfig(config: Map[String, String]): (String, Char, Boolean, String) = {

    assert(config.isDefinedAt("file"), "File name must be Defined!")
    val file: String = config("file")

    val delim: Char = if(config.isDefinedAt("delim")) {
      config("delim").toCharArray()(0)
    } else {
      ','
    }

    val head: Boolean = if(config.isDefinedAt("head")) {
      config("head") match {
        case "true" => true
        case "True" => true
        case "false" => false
        case "False" => false
      }
    } else {
      true
    }

    val task: String = if(config.isDefinedAt("task")) config("task") else ""

    (file, delim, head, task)
  }

  def apply(implicit config: Map[String, String]): GaussianLinearModel = {

    val (file, delim, head, task) = readConfig(config)
    val reader = utils.getCSVReader(file, delim)

    val graphconfig = Map("blueprints.graph" ->
      "com.tinkerpop.blueprints.impls.tg.TinkerGraph")

    val wMap: mutable.HashMap[String, AnyRef] = mutable.HashMap()
    val xMap: mutable.HashMap[Int, AnyRef] = mutable.HashMap()
    val yMap: mutable.HashMap[Int, AnyRef] = mutable.HashMap()
    val ceMap: mutable.HashMap[Int, AnyRef] = mutable.HashMap()
    val peMap: mutable.HashMap[Int, AnyRef] = mutable.HashMap()

    val fg = manager.create(GraphFactory.open(mapAsJavaMap(graphconfig)))

    var index = 1
    val (points, dim) = readCSV(reader, head)

    logger.log(Priority.INFO, "Creating graph for data set.")
    val pnode:Parameter = fg.addVertex(null, classOf[Parameter])
    pnode.setSlope(Array.fill[Double](dim)(1.0))
    wMap.put("w", pnode.asVertex().getId)

    points.foreach((couple) => {
      val xv = DenseVector.vertcat[Double](couple._1, DenseVector(Array(1.0)))
      val yv = couple._2
      /*
      * Create nodes xi and yi
      * append to them their values
      * properties, etc
      * */
      val xnode: Point = fg.addVertex(("x", index), classOf[Point])
      xnode.setValue(xv.toArray)
      xnode.setFeatureMap(xv.toArray)
      xMap.put(index, xnode.asVertex().getId)

      val ynode: Label = fg.addVertex(("y", index), classOf[Label])
      ynode.setValue(yv)
      yMap.put(index, ynode.asVertex().getId)

      //Add edge between xi and yi
      val ceEdge: CausalEdge = fg.addEdge((1, index), xnode.asVertex(),
        ynode.asVertex(), "causes",
        classOf[CausalEdge])
      ceEdge.setRelation("causal")
      ceMap.put(index, ceEdge.asEdge().getId)

      //Add edge between w and y_i
      val peEdge: ParamEdge = fg.addEdge((2, index), pnode.asVertex(),
        ynode.asVertex(), "controls", classOf[ParamEdge])
      peMap.put(index, peEdge.asEdge().getId)

      index += 1
    })

    val vMaps = (wMap, xMap, yMap)
    val eMaps = (ceMap, peMap)
    logger.log(Priority.INFO, "Graph constructed, now building model object.")
    new GaussianLinearModel(fg, index-1, dim, vMaps, eMaps, task).normalizeData
  }
}
