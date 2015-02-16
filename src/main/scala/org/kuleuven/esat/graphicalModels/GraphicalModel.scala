package org.kuleuven.esat.graphicalModels

import breeze.linalg._
import com.tinkerpop.blueprints.pgm.Edge
import org.kuleuven.esat.kernels.SVMKernel
import org.kuleuven.esat.optimization.Optimizer


/**
 * Basic Higher Level abstraction
 * for graphical models.
 *
 */
trait GraphicalModel[T] {
  protected val g: T
}

/**
 * Skeleton of Parameterized Graphical Model
 * @tparam G The type of the underlying graph.
 * @tparam K The type of indexing on the parameters
 * @tparam K2 The type of indexing on the feature vectors.
 * @tparam T The type of the parameters i.e. [[DenseVector]] or
 *           [[breeze.linalg.DenseMatrix]]
 * @tparam Q A Vector/Matrix representing the features of a point
 * @tparam R The type of the output of the predictive model
 *           i.e. A Real Number or a Vector of outputs.
 *
 * */
trait ParameterizedLearner[G, K, K2, T <: Tensor[K, Double], Q <: Tensor[K2, Double], R]
  extends GraphicalModel[G] {
  protected var params: T
  protected val optimizer: Optimizer[G, K, T, Q, R]
  protected val nPoints: Int

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   * */
  def learn(): Unit = {
    this.params = optimizer.optimize(this.g, nPoints, this.params,
      this.getParamOutEdges(), this.getxyPair)
  }

  /**
   * Get the value of the parameters
   * of the model.
   * */
  def parameters() = this.params

  def updateParameters(param: T): Unit = {
    this.params = param
  }

  def getParamOutEdges(): java.lang.Iterable[Edge]

  def getxyPair(ed: Edge): (Q, R)

}

/**
 * Represents skeleton of a
 * Generalized Linear Model.
 *
 * @tparam T The underlying type of the graph
 *           ex. Gremlin, Neo4j etc
 * @tparam K1 The type of indexing in the parameters
 * @tparam K2 The type of indexing in the feature space.
 * @tparam P A Vector/Matrix of Doubles indexed using [[K1]]
 * @tparam Q A Vector/Matrix representing the features of a point
 * @tparam R The type of the output of the predictive model
 *           i.e. A Real Number or a Vector of outputs.
 * */

abstract class LinearModel[T, K1, K2,
  P <: Tensor[K1, Double], Q <: Tensor[K2, Double], R]
  extends GraphicalModel[T]
  with ParameterizedLearner[T, K1, K2, P, Q, R] {

  var featureMap: (List[Q]) => List[Q] = (x) => x
  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: Q): R

  def getPredictors(): List[Q] = {
    val edges = this.getParamOutEdges.iterator()
    var res = List[Q]()
    while(edges.hasNext) {
      val (features, _) = this.getxyPair(edges.next())
      res = res.union(List(features))
    }
    res
  }

  def getTargets(): List[R] = {
    val edges = this.getParamOutEdges.iterator()
    var res = List[R]()
    while(edges.hasNext) {
      val (_, target) = this.getxyPair(edges.next())
      res = res.union(List(target))
    }
    res
  }

  def applyKernel(kernel: SVMKernel[DenseMatrix[Double]]): Unit = {}

}
