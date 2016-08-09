package io.github.mandar2812.dynaml.kernels

import breeze.linalg._
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.{Logger, Priority}

import scala.collection.immutable.HashMap
/**
 * Defines an abstract class outlines the basic
 * functionality requirements of an SVM Kernel
 */
abstract class SVMKernel[M] extends
CovarianceFunction[DenseVector[Double], Double, M]
with Serializable {

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
                    (data: DenseVector[Double])
  : DenseVector[Double] = {
    val kernel = DenseVector(prototypes.map((p) => this.evaluate(p, data)).toArray)
    val buff: Transpose[DenseVector[Double]] = kernel.t * decomposition._2
    val lambda: DenseVector[Double] = decomposition._1.map(lam => 1/math.sqrt(lam))
    val ans = buff.t
    ans :* lambda
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
  def buildSVMKernelMatrix[S <: Seq[T], T](
      mappedData: S,
      length: Int,
      eval: (T, T) =>  Double):
  KernelMatrix[DenseMatrix[Double]] = {

    logger.info("Constructing kernel matrix.")

    val kernelIndex = utils.combine(Seq(mappedData.zipWithIndex, mappedData.zipWithIndex))
      .map(s => ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
      .toMap

    val kernel = DenseMatrix.tabulate[Double](length, length){
      (i, j) => if (i >= j) kernelIndex((i,j)) else kernelIndex((j,i))
    }

    logger.info("Dimension: " + kernel.rows + " x " + kernel.cols)
    new SVMKernelMatrix(kernel, length)
  }

  def crossKernelMatrix[S <: Seq[T], T](data1: S, data2: S,
                                        eval: (T, T) =>  Double)
  : DenseMatrix[Double] = {

    logger.info("Constructing cross kernel matrix.")
    logger.info("Dimension: " + data1.length + " x " + data2.length)

    val kernelIndex = utils.combine(Seq(data1.zipWithIndex, data2.zipWithIndex))
      .map(s => ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
      .toMap

    DenseMatrix.tabulate[Double](data1.length, data2.length){
      (i, j) => kernelIndex((i,j))
    }
  }

}

/**
  * Kernels with a locally stored matrix in the form
  * of a breeze [[DenseMatrix]] instance.
  * */
trait LocalSVMKernel[Index] extends LocalScalarKernel[Index] {
  override def buildKernelMatrix[S <: Seq[Index]](
    mappedData: S,
    length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)
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
    logger.log(Priority.INFO, "Eigenvalue stats: "
      +min(decomp.eigenvalues)
      +" =< lambda =< "
      +max(decomp.eigenvalues)
    )
    (decomp.eigenvalues, decomp.eigenvectors)

  }

}

