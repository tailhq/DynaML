package org.kuleuven.esat.graphicalModels

import breeze.linalg._
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.{Graph, Edge}
import org.kuleuven.esat.evaluation.Metrics
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
  protected val optimizer: Optimizer[K, T, Q, R]
  protected val nPoints: Int
  def npoints = nPoints
  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   * */
  def learn(): Unit = {
    this.params = optimizer.optimize(nPoints, this.params,
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

  def setMaxIterations(i: Int): this.type = {
    this.optimizer.setNumIterations(i)
    this
  }

  def setBatchFraction(f: Double): this.type = {
    assert(f >= 0.0 && f <= 1.0, "Mini-Batch Fraction should be between 0.0 and 1.0")
    this.optimizer.setMiniBatchFraction(f)
    this
  }

  def setLearningRate(alpha: Double): this.type = {
    this.optimizer.setStepSize(alpha)
    this
  }

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
  with ParameterizedLearner[T, K1, K2, P, Q, R]
  with EvaluableModel[P, R] {

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: Q): R

  def clearParameters(): Unit

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
}

/**
 * An evaluable model is on in which
 * there is a function taking in a csv
 * reader object pointing to a test csv file
 * and returns the appropriate [[Metrics]] object
 *
 * @tparam P The type of the model's Parameters
 * @tparam R The type of the output value
 * */
trait EvaluableModel [P, R]{
  def evaluate(reader: CSVReader, head: Boolean): Metrics[R]
}

trait KernelizedModel[T <: Tensor[K1, Double], Q <: Tensor[K2, Double], R, K1, K2]
  extends LinearModel[Graph, K1, K2, T, Q, R]{

  /**
   * This variable stores the indexes of the
   * prototype points of the data set.
   * */
  protected var points: List[Int] = List()

  /**
   * The non linear feature mapping implicitly
   * defined by the kernel applied, this is initialized
   * to an identity map.
   * */
  var featureMap: (List[Q]) => List[Q] = (x) => x

  /**
   * Implements the changes in the model
   * after application of a given kernel.
   *
   * It calculates two things
   *
   * 1) Eigen spectrum of the kernel
   *
   * 2) Calculates an approximation to the
   * non linear feature map induced by the
   * application of the kernel
   *
   * @param kernel A kernel object.
   * @param M The number of prototypes to select
   *          in order to approximate the kernel
   *          matrix.
   * */
  def applyKernel(kernel: SVMKernel[DenseMatrix[Double]], M: Int): Unit = {}

  /**
   * Calculate an approximation to
   * the subset of size M
   * with the maximum entropy.
   * */
  def optimumSubset(M: Int): Unit

  /**
   * Apply the feature map calculated by
   * the using the Kernel to the data points
   * and store the mapped features in the respective
   * data nodes.
   * */
  def applyFeatureMap(): Unit

  /**
   * Tune the parameters of an RBF Kernel
   * so it best fits the data to be modeled.
   * */
  def tuneRBFKernel(implicit task: String): Unit

  /**
   * Cross validate the model on the
   * data set.
   * */
  def crossvalidate(folds: Int): (Double, Double, Double)

  /**
   * Get a subset of the data set defined
   * as a filter operation on the raw data set.
   *
   * @param fn A function which takes a data point
   *           and returns a boolean value.
   * @return The list containing all the data points
   *         satisfying the filtering criterion.
   * */
  def filter(fn : (Int) => Boolean): List[Q] =
    (1 to nPoints).view.filter(fn).map{
      i =>
        this.g.getVertex(("x", i))
          .getProperty("value")
          .asInstanceOf[Q]
    }.toList
}