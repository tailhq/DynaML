/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.kuleuven.esat.graphicalModels

import breeze.linalg.{DenseMatrix, norm, DenseVector}
import breeze.numerics.sigmoid
import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import org.apache.log4j.{Logger, Priority}
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.graphUtils.{Parameter, CausalEdge, Point}
import org.kuleuven.esat.kernels.{RBFKernel, SVMKernel, GaussianDensityKernel}
import org.kuleuven.esat.optimization.{ConjugateGradient, GradientDescent}
import org.kuleuven.esat.prototype.{QuadraticRenyiEntropy, GreedyEntropySelector}
import org.kuleuven.esat.utils
import scala.collection.JavaConversions
import scala.collection.mutable

/**
 * Abstract class implementing kernel feature map
 * extraction functions.
 */
abstract class KernelBayesianModel(implicit protected val task: String) extends
KernelizedModel[FramedGraph[Graph], Iterable[CausalEdge],
  DenseVector[Double], DenseVector[Double], Double, Int, Int] {

  protected val logger = Logger.getLogger(this.getClass)

  override protected val optimizer: ConjugateGradient

  protected val featuredims: Int

  protected val vertexMaps: (mutable.HashMap[String, AnyRef],
    mutable.HashMap[Int, AnyRef],
    mutable.HashMap[Int, AnyRef])

  protected val edgeMaps: (mutable.HashMap[Int, AnyRef],
    mutable.HashMap[Int, AnyRef])

  override def learn(): Unit = {
    this.params = optimizer.optimize(nPoints, this.params, this.getXYEdges())
  }

  def setRegParam(reg: Double): this.type = {
    this.optimizer.setRegParam(reg)
    this
  }

  override def getXYEdges() =
    JavaConversions.iterableAsScalaIterable(
      this.g.getEdges("relation", "causal", classOf[CausalEdge])
    )

  /**
   * Get a subset of the data set defined
   * as a filter operation on the raw data set.
   *
   * @param fn A function which takes a data point
   *           and returns a boolean value.
   * @return The list containing all the data points
   *         satisfying the filtering criterion.
   * */
  def filter(fn : (Int) => Boolean): List[DenseVector[Double]] =
    (1 to nPoints).view.filter(fn).map{
      i => {
        val point: Point = this.g.getVertex(vertexMaps._2(i),
          classOf[Point])
        DenseVector(point.getValue())
      }
    }.toList

  override def optimumSubset(M: Int): Unit = {
    points = (0 to this.npoints - 1).toList
    if(M < this.npoints) {
      logger.info("Calculating sample variance of the data set")

      //Get the original features of the data
      //Calculate the column means and variances
      val (mean, variance) = utils.getStats(this.filter((_) => true))

      //Use the adjusted value of the variance
      val adjvarance:DenseVector[Double] = variance :/= (npoints.toDouble - 1)
      val density = new GaussianDensityKernel

      logger.info("Using Silvermans rule of thumb to set bandwidth of density kernel")
      logger.info("Std Deviation of the data: "+adjvarance.toString())
      logger.info("norm: "+norm(adjvarance))
      density.setBandwidth(DenseVector.tabulate[Double](featuredims - 1){
        i => 1.06*math.sqrt(adjvarance(i))/math.pow(npoints, 0.2)
      })
      logger.info("Building low rank approximation to kernel matrix")

      points = GreedyEntropySelector.subsetSelection(this,
        M,
        new QuadraticRenyiEntropy(density),
        0.0001,
        2000)
    }
  }

  /**
   * Apply the feature map calculated by
   * the using the Kernel to the data points
   * and store the mapped features in the respective
   * data nodes.
   * */
  def applyFeatureMap(): Unit = {
    logger.info("Applying Feature map to data set")
    val edges = this.getXYEdges()
    val pnode:Parameter = this.g.getVertex(this.vertexMaps._1("w"),
      classOf[Parameter])
    pnode.setSlope(this.params.toArray)
    edges.foreach((edge) => {
      //Get the predictor vertex corresponding to the edge
      val vertex: Point = edge.getPoint()

      //Get the original features of the point
      val featurex = DenseVector(vertex.getValue())

      //Get mapped features for the point
      val mappedf = featureMap(List(featurex(0 to featurex.length - 2))).head
      val newFeatures = DenseVector.vertcat[Double](mappedf, DenseVector(1.0))
      //Set a new property in the vertex corresponding to the mapped features
      vertex.setFeatureMap(newFeatures.toArray)
    })
    logger.info("DONE: Applying Feature map to data set")
  }

  override def applyKernel(
      kernel: SVMKernel[DenseMatrix[Double]],
      M: Int = math.sqrt(npoints).toInt): Unit = {

    if(M != this.points.length) {
      this.optimumSubset(M)
    }

    val features_of_points = this.filter(p => this.points.contains(p))
    val kernelMatrix =
      kernel.buildKernelMatrix(features_of_points, M)
    val decomposition = kernelMatrix.eigenDecomposition(M)
    this.featureMap = kernel.featureMapping(decomposition)(features_of_points)
    this.params = DenseVector.ones[Double](decomposition._1.length + 1)
    this.applyFeatureMap()
  }

  def applyRBFKernel(
      kernel: RBFKernel,
      M: Int = math.sqrt(npoints).toInt): Unit = {
    this.featureMap = (points: List[DenseVector[Double]]) => {
      points.map((p) =>
      {
        p :/= kernel.getBandwidth
        val n = norm(p, 2)
        DenseVector.tabulate[Double](M){i =>
          val lambda = math.pow(math.pow(2, i)*utils.factorial(i)*math.sqrt(math.Pi), 0.5)
          math.exp(-n*n)*utils.hermite(i, n)/lambda
        }
      }
      )
    }
    this.params = DenseVector.ones[Double](M + 1)
    this.applyFeatureMap()
  }

  /**
   * Override the effect of appyling a kernel
   * and return the model back to its default
   * state i.e. the Identity Kernel
   * */
  override def clearParameters(): Unit = {
    this.params = DenseVector.ones[Double](this.featuredims)
    this.featureMap = (x) => x
    val it = this.getXYEdges()
    it.foreach((outEdge) => {
      val ynode = outEdge.getLabel()
      val xnode = outEdge.getPoint()
      xnode.setFeatureMap(xnode.getValue())
    })
    val paramNode: Parameter = this.g.getVertex(vertexMaps._1("w"),
      classOf[Parameter])
    paramNode.setSlope(this.params.toArray)
  }

  def crossvalidate(folds: Int = 10, reg: Double = 0.001): (Double, Double, Double) = {
    //Create the folds as lists of integers
    //which index the data points
    this.optimizer.setRegParam(reg).setNumIterations(5)
      .setStepSize(0.001).setMiniBatchFraction(1.0)
    var avg_metrics = DenseVector(0.0, 0.0, 0.0)
    for( a <- 1 to folds){
      //For the ath fold
      //partition the data
      //ceil(a-1*npoints/folds) -- ceil(a*npoints/folds)
      //as test and the rest as training

      val (test, train) = (1 to this.npoints).partition((p) =>
      {
        p >= (a-1)*this.nPoints/folds & p <= a*(this.nPoints-1)/folds
      })

      val training_data = train.map((p) => {
        val ed: CausalEdge = this.g.getEdge(this.edgeMaps._1(p),
          classOf[CausalEdge])
        ed
      }).view.toIterable

      val test_data = test.map((p) => {
        val ed: CausalEdge = this.g.getEdge(this.edgeMaps._1(p),
          classOf[CausalEdge])
        ed
      }).view.toIterable

      val tempparams = this.optimizer.optimize((folds - 1 / folds) * this.npoints,
        this.params,
        training_data)
      val metrics = KernelBayesianModel.evaluate(tempparams)(test_data)(this.task)
      val kpi = metrics.kpi()
      avg_metrics :+= kpi
    }
    //run batch sgd on each fold
    //and test
    (avg_metrics(0)/folds.toDouble,
      avg_metrics(1)/folds.toDouble,
      avg_metrics(2)/folds.toDouble)
  }

  def tuneRBFKernel(prot: Int = math.sqrt(this.nPoints.toDouble).toInt,
                    folds: Int,
                    task: String = this.task): Unit = {
    //Generate a grid of sigma values
    //val (samplemean, samplevariance) = utils.getStats(this.filter(_ => true))
    logger.info("Calculating grid for gamma values")
    //samplevariance :*= 1.0/(this.npoints.toDouble - 1)
    //val sigma = norm(samplevariance, 2)
    val sigmagrid = List.tabulate(30)((i) => 0.1+i.toDouble/10.0)

    val gammagrid = List.tabulate(30)((i) => i.toDouble/10.0)

    val grid = (for{s <- sigmagrid; g <- gammagrid} yield (s,g)).groupBy((c) => c._1).map((hyper) => {
      this.applyKernel(new RBFKernel(hyper._1), prot)
      hyper._2.map((sigmaAndGamma) => {
        logger.info("sigma = "+hyper._1+" gamma = "+hyper._2)
        val (a, b, c) = this.crossvalidate(folds, sigmaAndGamma._2)
        (c, sigmaAndGamma)
      })
    }).flatten
    logger.info("Grid: \n"+grid.toList)
    val max = grid.max
    logger.log(Priority.INFO, "Best value of sigma: "+max._2._1+" gamma: "+max._2._2)
    this.applyKernel(new RBFKernel(max._2._1), prot)
    this.setRegParam(max._2._2).setMaxIterations(5).setBatchFraction(1.0)
    this.learn()
  }
}

object KernelBayesianModel {
  val logger = Logger.getLogger(this.getClass)
  def evaluate(params: DenseVector[Double])
              (test_data_set: Iterable[CausalEdge])
              (task: String): Metrics[Double] = {
    var index: Int = 1
    val scoresAndLabels = test_data_set.map((e) => {
      val scorepred = GaussianLinearModel.score(params) _
      val x = DenseVector(e.getPoint().getFeatureMap())
      val y = e.getLabel().getValue()
      index += 1
      (scorepred(x(0 to x.length - 2)), y)
    })
    Metrics(task)(scoresAndLabels.toList, index)
  }
}