package io.github.mandar2812.dynaml.kernels

import breeze.linalg._
import io.github.mandar2812.dynaml.algebra.{PartitionedMatrix, PartitionedPSDMatrix}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
 * Defines an abstract class outlines the basic
 * functionality requirements of an SVM Kernel
 * */
trait SVMKernel[M] extends
CovarianceFunction[DenseVector[Double], Double, M]
with Serializable {

  /**
    * Builds an approximate nonlinear feature map
    * which corresponds to an SVM Kernel. This is
    * done using the Nystrom method i.e. approximating
    * the eigenvalues and eigenvectors of the Kernel
    * matrix of some data set.
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
    * */
  def featureMapping(decomposition: (DenseVector[Double], DenseMatrix[Double]))
                    (prototypes: List[DenseVector[Double]])
                    (data: DenseVector[Double])
  : DenseVector[Double] = {
    val kernel = DenseVector(prototypes.map((p) => this.evaluate(p, data)).toArray)
    val buff: Transpose[DenseVector[Double]] = kernel.t * decomposition._2
    val lambda: DenseVector[Double] = decomposition._1.map(lam => 1/math.sqrt(lam))
    val ans = buff.t
    ans *:* lambda
  }
}

/**
  * Defines a global singleton object [[SVMKernel]]
  * having functions which can construct kernel matrices.
  */
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

    val kernelIndex =
      utils.combine(Seq(mappedData.zipWithIndex, mappedData.zipWithIndex))
        .filter(s => s.head._2 >= s.last._2)
        .map(s => ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
        .toMap


    val kernel = DenseMatrix.tabulate[Double](length, length){
      (i, j) => if (i >= j) kernelIndex((i,j)) else kernelIndex((j,i))
    }

    println("   Dimensions: " + kernel.rows + " x " + kernel.cols)
    new SVMKernelMatrix(kernel, length)
  }

  def crossKernelMatrix[S <: Seq[T], T](data1: S, data2: S,
                                        eval: (T, T) =>  Double)
  : DenseMatrix[Double] = {

    val kernelIndex =
      utils.combine(Seq(data1.zipWithIndex, data2.zipWithIndex))
        .map(s => ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
        .toMap

    println("   Dimensions: " + data1.length + " x " + data2.length)
    DenseMatrix.tabulate[Double](data1.length, data2.length){
      (i, j) => kernelIndex((i,j))
    }
  }

  def buildKernelGradMatrix[S <: Seq[T], T](
    data1: S,
    hyper_parameters: Seq[String],
    eval: (T, T) => Double,
    evalGrad: String => (T, T) =>  Double):
  Map[String, DenseMatrix[Double]] = {

    val (rows, cols) = (data1.length, data1.length)
    println("Constructing Kernel/Grad Matrices")
    println("   Dimensions: " + rows + " x " + cols)

    val keys = Seq("kernel-matrix") ++ hyper_parameters

    utils.combine(Seq(data1.zipWithIndex, data1.zipWithIndex))
      .filter(s => s.head._2 >= s.last._2)
      .flatMap(s => {
        keys.map(k =>
          if(k == "kernel-matrix") (k, ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
          else (k, ((s.head._2, s.last._2), evalGrad(k)(s.head._1, s.last._1))))
      }).groupBy(_._1).map(cl => {

      if (cl._1 == "kernel-matrix") println("Constructing Kernel Matrix")
      else println("Constructing Grad Matrix for: "+cl._1)

      val kernelIndex = cl._2.map(_._2).toMap

      (
        cl._1,
        DenseMatrix.tabulate[Double](rows, cols){
          (i, j) => if (i >= j) kernelIndex((i,j)) else kernelIndex((j,i))
        }
      )
    })
  }


  /**
    * Returns the kernel matrix along with
    * its derivatives for each hyper-parameter.
    * */
  def buildCrossKernelGradMatrix[S <: Seq[T], T](
    data1: S, data2: S,
    hyper_parameters: Seq[String],
    eval: (T, T) => Double,
    evalGrad: (String) => (T, T) =>  Double):
  Map[String, DenseMatrix[Double]] = {

    val (rows, cols) = (data1.length, data2.length)
    println("Constructing Kernel/Grad Matrices")
    println("   Dimensions: " + rows + " x " + cols)

    val keys = Seq("kernel-matrix") ++ hyper_parameters

    utils.combine(Seq(data1.zipWithIndex, data2.zipWithIndex))
      .flatMap(s => {
        keys.map(k =>
          if(k == "kernel-matrix") (k, ((s.head._2, s.last._2), eval(s.head._1, s.last._1)))
          else (k, ((s.head._2, s.last._2), evalGrad(k)(s.head._1, s.last._1))))
      }).groupBy(_._1).map(cl => {

      if (cl._1 == "kernel-matrix") println("Constructing Kernel Matrix")
      else println("Constructing Grad Matrix for: "+cl._1)

      val kernelIndex = cl._2.map(_._2).toMap

      (
        cl._1,
        DenseMatrix.tabulate[Double](rows, cols){
          (i, j) => kernelIndex((i,j))
        }
      )
    })
  }

  def buildPartitionedKernelMatrix[S <: Seq[T], T](
    data: S,
    length: Long,
    numElementsPerRowBlock: Int,
    numElementsPerColBlock: Int,
    eval: (T, T) =>  Double): PartitionedPSDMatrix = {

    val (rows, cols) = (length, length)

    println("Constructing partitioned kernel matrix.")
    println("Dimension: " + rows + " x " + cols)

    val (num_R_blocks, num_C_blocks) = (
      math.ceil(rows.toDouble/numElementsPerRowBlock).toLong,
      math.ceil(cols.toDouble/numElementsPerColBlock).toLong)

    println("Blocks: " + num_R_blocks + " x " + num_C_blocks)
    val partitionedData = data.grouped(numElementsPerRowBlock).zipWithIndex.toStream

    println("~~~~~~~~~~~~~~~~~~~~~~~")
    println("Constructing Partitions")
    new PartitionedPSDMatrix(
      utils.combine(Seq(partitionedData, partitionedData))
        .filter(c => c.head._2 >= c.last._2)
        .toStream.map(c => {

        val partitionIndex = (c.head._2.toLong, c.last._2.toLong)
        println(":- Partition: "+partitionIndex)

        val matrix =
          if(partitionIndex._1 == partitionIndex._2)
            buildSVMKernelMatrix(c.head._1, c.head._1.length, eval).getKernelMatrix()
          else crossKernelMatrix(c.head._1, c.last._1, eval)

        (partitionIndex, matrix)
      })
    , rows, cols, num_R_blocks, num_C_blocks)

  }

  def crossPartitonedKernelMatrix[T, S <: Seq[T]](
    data1: S, data2: S,
    numElementsPerRowBlock: Int,
    numElementsPerColBlock: Int,
    eval: (T, T) => Double): PartitionedMatrix = {

    val (rows, cols) = (data1.length, data2.length)

    println("Constructing cross partitioned kernel matrix.")
    println("Dimension: " + rows + " x " + cols)

    val (num_R_blocks, num_C_blocks) = (
      math.ceil(rows.toDouble/numElementsPerRowBlock).toLong,
      math.ceil(cols.toDouble/numElementsPerColBlock).toLong)

    println("Blocks: " + num_R_blocks + " x " + num_C_blocks)
    println("~~~~~~~~~~~~~~~~~~~~~~~")
    println("Constructing Partitions")
    new PartitionedMatrix(utils.combine(Seq(
      data1.grouped(numElementsPerRowBlock).zipWithIndex.toStream,
      data2.grouped(numElementsPerColBlock).zipWithIndex.toStream)
    ).toStream.map(c => {
      val partitionIndex = (c.head._2.toLong, c.last._2.toLong)
      println(":- Partition: "+partitionIndex)
      val matrix = crossKernelMatrix(c.head._1, c.last._1, eval)
      (partitionIndex, matrix)
    }), rows, cols, num_R_blocks, num_C_blocks)
  }


  def buildPartitionedKernelGradMatrix[S <: Seq[T], T](
    data: S, length: Long,
    numElementsPerRowBlock: Int,
    numElementsPerColBlock: Int,
    hyper_parameters: Seq[String],
    eval: (T, T) => Double,
    evalGrad: (String) => (T, T) =>  Double): Map[String, PartitionedPSDMatrix] = {

    val (rows, cols) = (length, length)

    println("Constructing partitioned kernel matrix and its derivatives")
    println("Dimension: " + rows + " x " + cols)

    val (num_R_blocks, num_C_blocks) = (
      math.ceil(rows.toDouble/numElementsPerRowBlock).toLong,
      math.ceil(cols.toDouble/numElementsPerColBlock).toLong)

    println("Blocks: " + num_R_blocks + " x " + num_C_blocks)
    val partitionedData = data.grouped(numElementsPerRowBlock).zipWithIndex.toStream

    println("~~~~~~~~~~~~~~~~~~~~~~~")
    println("Constructing Partitions")


    //Build the result using flatMap - reduce
    utils.combine(Seq(partitionedData, partitionedData))
      .filter(c => c.head._2 >= c.last._2)
      .toStream.flatMap(c => {
      val partitionIndex = (c.head._2.toLong, c.last._2.toLong)
      print("\n")
      println(":- Partition: "+partitionIndex)

      if(partitionIndex._1 == partitionIndex._2) {
        SVMKernel.buildKernelGradMatrix(
          c.head._1,
          hyper_parameters,
          eval, evalGrad).map(cluster => {
          (cluster._1, (partitionIndex, cluster._2))
        }).toSeq

      } else {
        SVMKernel.buildCrossKernelGradMatrix(
          c.head._1, c.last._1,
          hyper_parameters,
          eval, evalGrad).map(cluster => {
          (cluster._1, (partitionIndex, cluster._2))
        }).toSeq

      }
    }).groupBy(_._1).map(cluster => {

      val hyp = cluster._1
      val matData = cluster._2.map(_._2)
      (hyp, new PartitionedPSDMatrix(matData, rows, cols, num_R_blocks, num_C_blocks))
    })

  }
}







