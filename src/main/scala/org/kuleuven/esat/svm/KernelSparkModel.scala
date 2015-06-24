package org.kuleuven.esat.svm

import breeze.linalg.{DenseMatrix, norm, DenseVector}
import breeze.numerics.sqrt
import org.apache.log4j.Logger
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.kuleuven.esat.graphicalModels.KernelizedModel
import org.kuleuven.esat.kernels.{SVMKernel, GaussianDensityKernel}
import org.kuleuven.esat.prototype.{QuadraticRenyiEntropy, GreedyEntropySelector}
import org.kuleuven.esat.utils

/**
 * Implementation of the Fixed Size
 * Kernel based LS SVM
 *
 * Fixed Size implies that the model
 * chooses a subset of the original
 * data to calculate a low rank approximation
 * to the kernel matrix.
 *
 * Feature Extraction is done in the primal
 * space using the Nystrom approximation.
 *
 * @author mandar2812
 */
abstract class KernelSparkModel(data: RDD[LabeledPoint], task: String)
  extends KernelizedModel[RDD[(Long, LabeledPoint)], RDD[LabeledPoint],
    DenseVector[Double], DenseVector[Double], Double, Int, Int]
  with Serializable {

  override protected val g = LSSVMSparkModel.indexedRDD(data)

  val colStats = Statistics.colStats(g.map(_._2.features))

  override protected val nPoints: Long = colStats.count

  override protected var hyper_parameters: List[String] = List("RegParam")

  override protected var current_state: Map[String, Double] = Map("RegParam" -> this.getRegParam)

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
  override def energy(h: Map[String, Double], options: Map[String, AnyRef]): Double = {
    //set the kernel paramters if options is defined
    //
    0.0
  }

  protected var featuredims: Int = g.first()._2.features.size

  protected var prototypes: List[DenseVector[Double]] = List()

  val logger = Logger.getLogger(this.getClass)

  override def getXYEdges: RDD[LabeledPoint] = data

  def getRegParam: Double

  override def optimumSubset(M: Int): Unit = {
    points = (0L to this.npoints - 1).toList
    if (M < this.npoints) {
      logger.info("Calculating sample variance of the data set")

      //Get the original features of the data
      //Calculate the column means and variances
      val (mean, variance) = (DenseVector(colStats.mean.toArray),
        DenseVector(colStats.variance.toArray))

      //Use the adjusted value of the variance
      val adjvarance: DenseVector[Double] = variance :/= (npoints.toDouble - 1)
      val density = new GaussianDensityKernel

      logger.info("Using Silvermans rule of thumb to set bandwidth of density kernel")
      logger.info("Std Deviation of the data: " + adjvarance.toString())
      logger.info("norm: " + norm(adjvarance))
      density.setBandwidth(DenseVector.tabulate[Double](featuredims) {
        i => 1.06 * math.sqrt(adjvarance(i)) / math.pow(npoints, 0.2)
      })
      logger.info("Building low rank approximation to kernel matrix")

      prototypes = GreedyEntropySelector.subsetSelection(this.g,
        new QuadraticRenyiEntropy(density), M, 25, 0.0001)
        .map(_._2.features.toArray).collect()
        .map(DenseVector(_)).toList
    }
  }

  override def applyKernel(kernel: SVMKernel[DenseMatrix[Double]],
                           M: Int = math.sqrt(npoints).toInt): Unit = {

    if(M != this.prototypes.length) {
      this.optimumSubset(M)
    }

    val (mean, variance) = (DenseVector(colStats.mean.toArray),
      DenseVector(colStats.variance.toArray))

    val scalingFunc = KernelSparkModel.scalePrototype(mean, variance) _

    val scaledPrototypes = prototypes map scalingFunc

    val kernelMatrix =
      kernel.buildKernelMatrix(scaledPrototypes, M)
    val decomposition = kernelMatrix.eigenDecomposition(M)

    var selectedEigenVectors: List[DenseMatrix[Double]] = List()
    var selectedEigenvalues: List[Double] = List()

    (0 until M).foreach((p) => {
      //Check the Girolami criterion
      // (1.u)^2 >= 2M/(1+M)
      //This increases parsimony
      val u = decomposition._2(::, p)
      if(math.pow(norm(u,1), 2.0) >= 2.0*M/(1.0+M.toDouble)) {
        selectedEigenvalues :+= decomposition._1(p)
        selectedEigenVectors :+= u.toDenseMatrix
      }

    })
    logger.info("Selected Components: "+selectedEigenvalues.length)
    val decomp = (DenseVector(selectedEigenvalues.toArray),
      DenseMatrix.vertcat(selectedEigenVectors:_*).t)

    this.featureMap = kernel.featureMapping(decomp)(scaledPrototypes)
    this.params = DenseVector.ones[Double](decomp._1.length + 1)
  }

}

object KernelSparkModel {
  def scalePrototype(mean: DenseVector[Double],
                     variance: DenseVector[Double])
                    (prototype: DenseVector[Double]): DenseVector[Double] =
    (prototype - mean)/sqrt(variance)
}
