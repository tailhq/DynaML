package org.kuleuven.esat.kernels

import breeze.linalg.{eig, DenseVector, DenseMatrix}
import org.apache.log4j.{Priority, Logger}
/**
 * Defines an abstract class outlines the basic
 * functionality requirements of an SVM Kernel
 */
abstract class SVMKernel[T] extends Kernel with Serializable {

  private var REGULARIZER: Double = 0.00

  /**
   * Build the kernel matrix of the prototype vectors
   *
   * @param mappedData The prototype vectors/points
   *
   * @param length The number of points
   *
   * @return A [[KernelMatrix]] object
   *
   *
   **/
  def buildKernelMatrix(
      mappedData: List[DenseVector[Double]],
      length: Int): KernelMatrix[T]

  /**
   * Builds an approximate nonlinear feature map
   * which corresponds to an SVM Kernel. This is
   * done using the Nystrom method i.e. approximating
   * the eigenvalues and eigenvectors of the Kernel
   * matrix of a given RDD
   *
   * For each data point,
   * calculate m dimensions of the
   * feature map where m is the number
   * of eigenvalues/vectors obtained from
   * the Eigen Decomposition.
   *
   * phi_i(x) = (1/sqrt(eigenvalue(i)))*Sum(k, 1, m, K(k, x)*eigenvector(i)(k))
   *
   * @param decomposition The Eigenvalue decomposition calculated
   *                      from the kernel matrix of the prototype
   *                      subset.
   * @param prototypes The prototype subset.
   *
   * @param data  The dataset on which the feature map
   *              is to be applied.
   *
   **/
  def featureMapping(decomposition: (DenseVector[Double], DenseMatrix[Double]))
                    (prototypes: List[DenseVector[Double]])
                    (data: List[DenseVector[Double]])
  : List[DenseVector[Double]] = {

    data.map((point) => DenseVector.tabulate(decomposition._1.length) { (i) =>
      val eigenvalue = if(decomposition._1(i) != Double.NaN) decomposition._1(i) else 0.0
      val eigenvector = decomposition._2(::, i).map{i => if(i == Double.NaN) 0.0 else i}
      var sum: Double = 0
      prototypes.foreach((prototype) =>
        sum += (1 / (REGULARIZER + math.sqrt(eigenvalue))) *
          this.evaluate(prototype, point)
      )
      sum
    })
  }
}

/**
 * Defines a global singleton object
 * [[SVMKernel]] which has useful functions
 *
 * */
object SVMKernel {

  private val logger = Logger.getLogger(this.getClass)

  /**
   * This function constructs an [[SVMKernelMatrix]]
   *
   * @param mappedData Graphical model
   * @param length Number of data points
   * @param eval A function which calculates the value of the Kernel
   *             given two feature vectors.
   *
   * @return An [[SVMKernelMatrix]] object.
   *
   * */
  def buildSVMKernelMatrix(
      mappedData: List[DenseVector[Double]],
      length: Int,
      eval: (DenseVector[Double], DenseVector[Double]) =>  Double):
  KernelMatrix[DenseMatrix[Double]] = {

    logger.log(Priority.INFO, "Constructing key-value representation of kernel matrix.")
    logger.log(Priority.INFO, "Dimension: " + length + " x " + length)

    val kernel = DenseMatrix.tabulate[Double](length, length){
      (i, j) => eval(mappedData(i), mappedData(j))
    }
    new SVMKernelMatrix(kernel, length)
  }

}

/**
 * Defines a trait which outlines the basic
 * functionality of Kernel Matrices.
 * */
trait KernelMatrix[T] extends Serializable {
  protected val kernel: T

  def eigenDecomposition(dimensions: Int): (DenseVector[Double], DenseMatrix[Double])

  def getKernelMatrix(): T = this.kernel
}

class SVMKernelMatrix(
    override protected val kernel: DenseMatrix[Double],
    private val dimension: Long)
  extends KernelMatrix[DenseMatrix[Double]]
  with Serializable {
  private val logger = Logger.getLogger(this.getClass)

  /**
   * Calculates the approximate eigen-decomposition of the
   * kernel matrix
   *
   * @param dimensions The effective number of dimensions
   *                   to be calculated in the feature map
   *
   * @return A Scala [[Tuple2]] containing the eigenvalues
   *         and eigenvectors.
   *
   * */
  override def eigenDecomposition(dimensions: Int = this.dimension.toInt):
  (DenseVector[Double], DenseMatrix[Double]) = {
    logger.log(Priority.INFO, "Eigenvalue decomposition of the kernel matrix using JBlas.")
    val decomp = eig(this.kernel)
    //TODO: Complete the code below
    /*val threshold:Double = 2*dimension/(dimension+1)
    var effectiveEigs = DenseMatrix.ones[Double](dimensions, dimensions)

    for(i <- 0 to dimensions-1) {

    }*/
    (decomp.eigenvalues, decomp.eigenvectors)
  }

}

