package io.github.mandar2812.dynaml.kernels

import breeze.linalg._
import io.github.mandar2812.dynaml.analysis.{KernelMatrix, SVMKernelMatrix}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
 * Defines an abstract class outlines the basic
 * functionality requirements of an SVM Kernel
 */
trait SVMKernel[M] extends
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
      .filter(s => s.head._2 >= s.last._2)
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







