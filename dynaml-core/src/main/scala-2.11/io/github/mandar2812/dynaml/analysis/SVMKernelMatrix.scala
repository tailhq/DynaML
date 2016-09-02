package io.github.mandar2812.dynaml.analysis

import breeze.linalg.{DenseMatrix, DenseVector, eig, max, min}
import org.apache.log4j.{Logger, Priority}

/**
  * @author mandar .
  */
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
