package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.models.StochasticProcessMixtureModel
import io.github.mandar2812.dynaml.probability.{ContMixtureRVBars, ContinuousDistrMixture}
import io.github.mandar2812.dynaml.probability.distributions.BlockedMultiVariateGaussian
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * Represents a multinomial mixture of GP models
  * @tparam I The index set (input domain) over which each component GP is
  *           defined.
  *
  * @author mandar2812 date 14/06/2017.
  * */
class GaussianProcessMixture[I: ClassTag](
  val component_processes: Seq[AbstractGPRegressionModel[_, I]],
  val weights: DenseVector[Double]) extends
  StochasticProcessMixtureModel[
    Seq[(I, Double)], I, Double, ContMixtureRVBars[
    PartitionedVector, PartitionedPSDMatrix, BlockedMultiVariateGaussian]] {

  private val logger = Logger.getLogger(this.getClass)


  protected val blockSize: Int = component_processes.head._blockSize

  /**
    *
    * Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * */
  override def predictiveDistribution[U <: Seq[I]](test: U) =
    ContinuousDistrMixture(blockSize)(
      component_processes.map(_.predictiveDistribution(test).underlyingDist),
      weights)


  /**
    * The training data
    * */
  override protected val g: Seq[(I, Double)] = Seq()

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    * */
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int) = {

    val posterior = predictiveDistribution(testData)

    val mean = posterior.underlyingDist.mean.toStream

    val (lower, upper) = posterior.underlyingDist.confidenceInterval(sigma.toDouble)

    val lowerErrorBars = lower.toStream
    val upperErrorBars = upper.toStream

    logger.info("Generating error bars")

    val preds = mean.zip(lowerErrorBars.zip(upperErrorBars)).map(t => (t._1, t._2._1, t._2._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }


  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    * */
  override def dataAsSeq(data: Seq[(I, Double)]) = data

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * */
  override def predict(point: I) = predictionWithErrorBars(Seq(point), 1).head._2
}
