package io.github.mandar2812.dynaml.models.svm

import breeze.linalg.{DenseMatrix, DenseVector}
import com.tinkerpop.blueprints.{Graph, GraphFactory}
import com.tinkerpop.frames.{FramedGraph, FramedGraphFactory}
import io.github.mandar2812.dynaml.evaluation.Metrics
import io.github.mandar2812.dynaml.graph._
import io.github.mandar2812.dynaml.graph.utils._
import io.github.mandar2812.dynaml.kernels.SVMKernel
import io.github.mandar2812.dynaml.models.SubsampledDualLSSVM
import io.github.mandar2812.dynaml.optimization.ConjugateGradient
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.{Logger, Priority}

import scala.collection.JavaConversions._
import scala.collection.mutable

class SDLSSVMModel(override protected val g: FramedGraph[Graph],
                   override protected val nPoints: Long,
                   override protected val featuredims: Int,
                   override protected val vertexMaps: (mutable.HashMap[String, AnyRef],
                     mutable.HashMap[Long, AnyRef],
                     mutable.HashMap[Long, AnyRef]),
                   override protected val edgeMaps: (mutable.HashMap[Long, AnyRef],
                     mutable.HashMap[Long, AnyRef]),
                   override implicit protected val task: String)
  extends LSSVMModel(g, nPoints, featuredims, vertexMaps, edgeMaps, task) with
  SubsampledDualLSSVM[FramedGraph[Graph], Iterable[CausalEdge]]{


  override protected var effectivedims = featuredims

  override def evaluateFold(params: DenseVector[Double])
                           (test_data_set: Iterable[CausalEdge])
                           (task: String): Metrics[Double] = {
    var index: Int = 1
    val prototypes = this.filterFeatures(p => this.points.contains(p))

    val scoresAndLabels = test_data_set.map((e) => {

      val x = DenseVector(e.getPoint().getFeatureMap())
      val y = e.getLabel().getValue()
      index += 1
      (score(x(0 to -2)), y)
    })
    Metrics(task)(scoresAndLabels.toList, index)
  }

  override def crossvalidate(folds: Int = 10, reg: Double = 0.001,
                             optionalStateFlag: Boolean = false): (Double, Double, Double) = {
    //Create the folds as lists of integers
    //which index the data points


    if(optionalStateFlag || feature_a == null) {
      val featuremats = SDLSSVMModel.getFeatureMatrix(points.length.toLong, kernel,
        getXYEdges().filter((p) => this.points.contains(p)), this.initParams(),
        1.0, reg)

      feature_a = featuremats._1
      b = featuremats._2
    }

    this.optimizer.setRegParam(reg).setNumIterations(this.params.length)
      .setStepSize(0.001).setMiniBatchFraction(1.0)

    val params = ConjugateGradient.runCG(feature_a, b,
      this.initParams(), 0.0001,
      this.params.length)
    val metrics =
      this.evaluateFold(params)(getXYEdges().filter((p) => !this.points.contains(p)))(this.task)
    val ans = metrics.kpi()
    (ans(0), ans(1), ans(2))
  }

  override def score(point: DenseVector[Double]): Double = {
    val rescaled = rescale(point)
    val prototypes = this.filterFeatures(p => this.points.contains(p))
    params dot DenseVector(prototypes.map(p => this.kernel.evaluate(p, rescaled)).toArray :+ 1.0)
  }
}

object SDLSSVMModel {

  val manager: FramedGraphFactory = new FramedGraphFactory
  val logger = Logger.getLogger(this.getClass)

  def getArgs(implicit config: Map[String, String]) = {
    val (file, delim, head, task) = LSSVMModel.readConfig(config)
    val reader = utils.getCSVReader(file, delim)

    val graphconfig = Map("blueprints.graph" ->
      "com.tinkerpop.blueprints.impls.tg.TinkerGraph")

    val wMap: mutable.HashMap[String, AnyRef] = mutable.HashMap()
    val xMap: mutable.HashMap[Long, AnyRef] = mutable.HashMap()
    val yMap: mutable.HashMap[Long, AnyRef] = mutable.HashMap()
    val ceMap: mutable.HashMap[Long, AnyRef] = mutable.HashMap()
    val peMap: mutable.HashMap[Long, AnyRef] = mutable.HashMap()

    val fg = manager.create(GraphFactory.open(mapAsJavaMap(graphconfig)))

    var index = 1
    val (points, dim) = LSSVMModel.readCSV(reader, head)

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

    (fg, index-1, dim, vMaps, eMaps, task)
  }

  def getFeatureMatrix(nPoints: Long,
                       kernel: SVMKernel[DenseMatrix[Double]],
                       ParamOutEdges: Iterable[CausalEdge],
                       initialP: DenseVector[Double],
                       frac: Double, regParam: Double)
  : (DenseMatrix[Double], DenseVector[Double]) = {

    val kernelmat = kernel.buildKernelMatrix(
      ParamOutEdges.map(p => {
        val vec = DenseVector(p.getPoint().getFeatureMap())
        vec(0 to -2)
      }).toList,
      nPoints.toInt)
      .getKernelMatrix()

    val smoother = DenseMatrix.eye[Double](nPoints.toInt)/regParam

    val ones = DenseMatrix.fill[Double](1,nPoints.toInt)(1.0)
    val y: DenseVector[Double] = DenseVector(ParamOutEdges.map(p => p.getLabel().getValue()).toArray)
    /**
      * A = [K + I/reg]|[1]
      *     [1.t]      |[0]
      * */
    val A = DenseMatrix.horzcat(
      DenseMatrix.vertcat(kernelmat + smoother, ones),
      DenseMatrix.vertcat(ones.t, DenseMatrix(0.0))
    )

    val b = DenseVector.vertcat(y, DenseVector(0.0))
    (A,b)
  }

  def apply(implicit config: Map[String, String]): SDLSSVMModel = {
    val (fg, index, dim, vMaps, eMaps, task) = getArgs(config)
    new SDLSSVMModel(fg, index-1, dim, vMaps, eMaps, task).normalizeData
  }

}