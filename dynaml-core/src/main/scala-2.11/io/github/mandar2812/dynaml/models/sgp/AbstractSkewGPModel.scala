package io.github.mandar2812.dynaml.models.sgp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.BlockedMESNRV
import io.github.mandar2812.dynaml.probability.distributions.BlockedMESN
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * Created by mandar on 28/02/2017.
  */
abstract class AbstractSkewGPModel[T, I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int, lambda: Double, tau: Double,
  meanFunc: DataPipe[I, Double] = DataPipe((_:I) => 0.0))
  extends ContinuousProcessModel[T, I, Double, BlockedMESNRV]
    with SecondOrderProcessModel[T, I, Double, Double, DenseMatrix[Double], BlockedMESNRV]
    with GloballyOptimizable {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * The training data
    **/
  override protected val g: T = data

  /**
    * Mean Function: Takes a member of the index set (input)
    * and returns the corresponding mean of the distribution
    * corresponding to input.
    **/
  override val mean = meanFunc
  /**
    * Underlying covariance function of the
    * Gaussian Processes.
    **/
  override val covariance = cov

  val noiseModel = n


  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters ++ List("skewness", "cutoff")


  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state ++ Map("skewness" -> lambda, "cutoff" -> tau)


  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {

    val (covHyp, noiseHyp) = (
      s.filterKeys(covariance.hyper_parameters.contains),
      s.filterKeys(noiseModel.hyper_parameters.contains)
    )

    covariance.setHyperParameters(covHyp)
    noiseModel.setHyperParameters(noiseHyp)

    current_state = covariance.state ++ noiseModel.state
    current_state += ("skewness" -> s("skewness"), "cutoff" -> s("cutoff"))
    this
  }


  val npoints = num

  protected var blockSize = 1000

  protected lazy val trainingData: Seq[I] = dataAsIndexSeq(g)

  protected lazy val trainingDataLabels = PartitionedVector(
    dataAsSeq(g).toStream.map(_._2),
    trainingData.length.toLong, _blockSize
  )

  def blockSize_(b: Int): Unit = {
    blockSize = b
  }

  def _blockSize: Int = blockSize

  protected var caching: Boolean = false

  protected var partitionedKernelMatrixCache: PartitionedPSDMatrix = _


  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U) = {
    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data


    val (l,t) = (current_state("skewness"), current_state("cutoff"))

    val priorMeanTest = PartitionedVector(
      test.map(mean(_))
        .grouped(_blockSize)
        .zipWithIndex.map(c => (c._2.toLong, DenseVector(c._1.toArray)))
        .toStream,
      test.length.toLong)

    val trainingMean = PartitionedVector(
      dataAsSeq(g).toStream.map(_._1).map(mean(_)),
      trainingData.length.toLong, _blockSize
    )

    val priorSkewnessTest = priorMeanTest.map(b => (b._1, DenseVector.fill[Double](b._2.length)(l)))

    val skewnessTraining = trainingMean.map(b => (b._1, DenseVector.fill[Double](b._2.length)(l)))

    val priorCutoff = current_state("cutoff")

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    val smoothingMat = if(!caching) {
      logger.info("---------------------------------------------------------------")
      logger.info("Calculating covariance matrix for training points")
      SVMKernel.buildPartitionedKernelMatrix(trainingData,
        trainingData.length, _blockSize, _blockSize,
        effectiveTrainingKernel.evaluate)
    } else {
      logger.info("** Using cached training matrix **")
      partitionedKernelMatrixCache
    }

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix for test points")
    val kernelTest = SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize, covariance.evaluate)

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix between training and test points")
    val crossKernel = SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      covariance.evaluate)

    val (predMean, predCov, predSkewness, predCutoff) = AbstractSkewGPModel.solveSkewGP(
      trainingDataLabels, trainingMean, priorMeanTest,
      smoothingMat, kernelTest, crossKernel,
      skewnessTraining, priorSkewnessTest,
      priorCutoff)

    BlockedMESNRV(predCutoff, predSkewness, predMean, predCov)
  }


  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  override def persist(state: Map[String, Double]): Unit = {
    setState(state)
    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))


    partitionedKernelMatrixCache = SVMKernel.buildPartitionedKernelMatrix(trainingData,
      trainingData.length, _blockSize, _blockSize,
      effectiveTrainingKernel.evaluate)
    caching = true

  }

  /**
    * Forget the cached kernel & noise matrices.
    * */
  def unpersist(): Unit = {
    partitionedKernelMatrixCache = null
    caching = false
  }


}


object AbstractSkewGPModel {

  def solveSkewGP(
    trainingLabels: PartitionedVector,
    trainingMean: PartitionedVector,
    priorMeanTest: PartitionedVector,
    smoothingMat: PartitionedPSDMatrix,
    kernelTest: PartitionedPSDMatrix,
    crossKernel: PartitionedMatrix,
    skewnessTraining: PartitionedVector,
    priorSkewnessTest: PartitionedVector,
    priorCutoff: Double): (PartitionedVector, PartitionedPSDMatrix, PartitionedVector, Double) = {

    val Lmat: LowerTriPartitionedMatrix = bcholesky(smoothingMat)

    val alpha: PartitionedVector = Lmat.t \\ (Lmat \\ (trainingLabels-trainingMean))

    val beta : PartitionedVector = Lmat.t \\ (Lmat \\ skewnessTraining)

    val delta: Double = 1.0/sqrt(1.0 + (skewnessTraining dot beta))

    val v: PartitionedMatrix = Lmat \\ crossKernel

    val varianceReducer: PartitionedMatrix = v.t * v

    //Ensure that the variance reduction is symmetric
    val adjustedVarReducer: PartitionedMatrix = (varianceReducer.L + varianceReducer.L.t).map(bm =>
      if(bm._1._1 == bm._1._2) (bm._1, bm._2*(DenseMatrix.eye[Double](bm._2.rows)*0.5))
      else bm)

    val reducedVariance: PartitionedPSDMatrix =
      new PartitionedPSDMatrix(
        (kernelTest - adjustedVarReducer).filterBlocks(c => c._1 >= c._2),
        kernelTest.rows, kernelTest.cols)

    (
      priorMeanTest + crossKernel.t * alpha,
      reducedVariance,
      (priorSkewnessTest - crossKernel.t * beta)*delta,
      (priorCutoff + (skewnessTraining dot alpha))*delta)

  }
}