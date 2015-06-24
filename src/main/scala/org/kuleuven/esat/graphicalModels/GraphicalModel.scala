package org.kuleuven.esat.graphicalModels

import breeze.linalg._
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.kernels._
import org.kuleuven.esat.optimization.{GloballyOptimizable, Optimizer}

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
 * @tparam K The type of indexing on the feature vectors.
 * @tparam T The type of the parameters
 * @tparam Q A Vector/Matrix representing the features of a point
 * @tparam R The type of the output of the predictive model
 *           i.e. A Real Number or a Vector of outputs.
 * @tparam S The type of the edge containing the
 *           features and label.
 *
 * */
trait ParameterizedLearner[G, K, T, Q <: Tensor[K, Double], R, S]
  extends GraphicalModel[G] {
  protected var params: T
  protected val optimizer: Optimizer[K, T, Q, R, S]
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
 * @tparam T The underlying type of the data structure
 *           ex. Gremlin, Neo4j, Spark RDD etc
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
  with ParameterizedLearner[T, K2, P, Q, R, S]
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

abstract class KernelizedModel[G, L, T <: Tensor[K1, Double], Q <: Tensor[K2, Double], R, K1, K2]
  extends LinearModel[G, K1, K2, T, Q, R, L] with GloballyOptimizable {

  protected val nPoints: Long

  def npoints = nPoints

  /**
   * This variable stores the indexes of the
   * prototype points of the data set.
   * */
  protected var points: List[Long] = List()

  /**
   * The non linear feature mapping implicitly
   * defined by the kernel applied, this is initialized
   * to an identity map.
   * */
  var featureMap: (Q) => Q = identity

  def getXYEdges: L

  /**
   * Implements the changes in the model
   * after application of a given kernel.
   *
   * It calculates
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
  def applyKernel(kernel: SVMKernel[DenseMatrix[Double]],
                  M: Int = math.sqrt(nPoints).toInt): Unit = {}

  /**
   * Calculate an approximation to
   * the subset of size M
   * with the maximum entropy.
   * */
  def optimumSubset(M: Int): Unit

  def crossvalidate(folds: Int, reg: Double): (Double, Double, Double)

  /**
   * Calculates the energy of the configuration,
   * in most global optimization algorithms
   * we aim to find an approximate value of
   * the hyper-parameters such that this function
   * is minimized.
   *
   * @param h The value of the hyper-parameters in the configuration space
   * @param options Optional parameters about configuration
   * @return Configuration Energy E(h)
   **/
  override def energy(h: Map[String, Double], options: Map[String, String]): Double = {
    //set the kernel paramters if options is defined
    //then set model parameters and cross validate

    if(options.contains("kernel")) {
      val kern = options("kernel") match {
        case "RBF" => new RBFKernel(1.0).setHyperParameters(h)
        case "Polynomial" => new PolynomialKernel(2, 1.0).setHyperParameters(h)
        case "Exponential" => new ExponentialKernel(1.0).setHyperParameters(h)
        case "Laplacian" => new LaplacianKernel(1.0).setHyperParameters(h)
      }
      //check if h and this.current_state have the same kernel params
      //calculate kernParam(h)
      //calculate kernParam(current_state)
      //if both differ in any way then apply
      //the kernel
      val kernh = h.filter((couple) => kern.hyper_parameters.contains(couple._1))
      val kerncs = current_state.filter((couple) => kern.hyper_parameters.contains(couple._1))
      if(!(kernh sameElements kerncs)) {
        this.applyKernel(kern)
      }
    }
    current_state = h
    val (_,_,e) = this.crossvalidate(4, h("RegParam"))

    1.0-e
  }

}