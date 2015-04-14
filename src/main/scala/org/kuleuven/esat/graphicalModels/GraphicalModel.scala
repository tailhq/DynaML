package org.kuleuven.esat.graphicalModels

import breeze.linalg._
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.blueprints.{Graph, Edge}
import com.tinkerpop.frames.{EdgeFrame, FramedGraph}
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.graphUtils.{CausalEdge, Point}
import org.kuleuven.esat.kernels.SVMKernel
import org.kuleuven.esat.optimization.Optimizer
import scala.pickling._
import binary._
import scala.collection.mutable


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
 * @tparam S The type of the edge containing the
 *           features and label.
 *
 * */
trait ParameterizedLearner[G, K, K2, T <: Tensor[K, Double],
Q <: Tensor[K2, Double], R, S]
  extends GraphicalModel[G] {
  protected var params: T
  protected val optimizer: Optimizer[K, T, Q, R, S]
  protected val nPoints: Int
  def npoints = nPoints
  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   * */
  def learn(): Unit

  /**
   * Get the value of the parameters
   * of the model.
   * */
  def parameters() = this.params

  def updateParameters(param: T): Unit = {
    this.params = param
  }

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
 * @tparam S The type of the edge containing the
 *           features and label.
 * */

abstract class LinearModel[T, K1, K2,
  P <: Tensor[K1, Double], Q <: Tensor[K2, Double], R, S]
  extends GraphicalModel[T]
  with ParameterizedLearner[T, K1, K2, P, Q, R, S]
  with EvaluableModel[P, R] {

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   * */
  def predict(point: Q): R

  def clearParameters(): Unit

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
  def evaluate(config: Map[String, String]): Metrics[R]
}

trait KernelizedModel[T <: Tensor[K1, Double], Q <: Tensor[K2, Double], R, K1, K2]
  extends LinearModel[FramedGraph[Graph], K1, K2, T, Q, R, CausalEdge[Array[Byte]]]{

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

  protected val vertexMaps: (mutable.HashMap[String, AnyRef],
    mutable.HashMap[Int, AnyRef],
    mutable.HashMap[Int, AnyRef])

  protected val edgeMaps: (mutable.HashMap[Int, AnyRef],
    mutable.HashMap[Int, AnyRef])

  def getXYEdges(): Iterable[CausalEdge[Array[Byte]]]

  override def learn(): Unit = {
    this.params = optimizer.optimize(nPoints, this.params,
      this.getXYEdges())
  }

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
  def filter(fn : (Int) => Boolean): List[Q]
}