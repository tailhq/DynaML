package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.models.StochasticProcessMixtureModel
import io.github.mandar2812.dynaml.probability.{ContinuousDistrMixture, MultGaussianPRV}
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
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
    Seq[(I, Double)], I, Double,
    ContinuousDistrMixture[
      PartitionedVector,
      MultGaussianPRV]] {

  private val logger = Logger.getLogger(this.getClass)

  /**
    *
    * Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * */
  override def predictiveDistribution[U <: Seq[I]](test: U) =
    ContinuousDistrMixture[PartitionedVector, MultGaussianPRV](
      component_processes.map(_.predictiveDistribution(test)),
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

    val posterior_components = component_processes.map(_.predictiveDistribution(testData))

    val post_means = posterior_components.map(_.mu)

    //._data.map(_._2.toArray.toStream).reduceLeft((a, b) => a ++ b)

    val error_bars_components = posterior_components.map(
      _.underlyingDist.confidenceInterval(sigma.toDouble)
    )

    val weightsArr = weights.toArray

    val mean = post_means.zip(weightsArr).map(c => c._1*c._2).reduce((a, b) => a+b).toStream

    val combined_error_bars_vec = error_bars_components.zip(weightsArr)
      .map(c => (c._1._1*c._2,c._1._2*c._2))
      .reduce((a,b) => (a._1+b._1, a._2+b._2))

    val (lowerErrorBars, upperErrorBars) = (
      combined_error_bars_vec._1.toStream,
      combined_error_bars_vec._2.toStream)


    logger.info("Generating error bars")
    //val preds = (mean zip stdDev).map(j => (j._1, j._1 - sigma*j._2, j._1 + sigma*j._2))
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
