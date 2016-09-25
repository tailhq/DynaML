package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.DenseVector
import breeze.stats.distributions._

/**
  * Created by mandar on 25/09/2016.
  */
case class MultivariateUniform(low: DenseVector[Double], high: DenseVector[Double])(implicit rand: RandBasis = Rand)
  extends ContinuousDistr[DenseVector[Double]] with Moments[DenseVector[Double], DenseVector[Double]] {

  assert(low.length == high.length, "Number of dimensions in lower and upper limit vectors must match!")
  assert((0 to low.length).forall(index => low(index) < high(index)),
    "Lower limit must be actually lesser than upper limit")

  val marginalDistributions: Array[Uniform] = Array.tabulate[Uniform](low.length)(i => new Uniform(low(i), high(i)))

  override def unnormalizedLogPdf(x: DenseVector[Double]): Double =
    x.toArray.zip(marginalDistributions).map(c => c._2.unnormalizedLogPdf(c._1)).sum

  override def logNormalizer: Double = marginalDistributions.map(c => c.logNormalizer).sum

  override def mean: DenseVector[Double] = DenseVector(marginalDistributions.map(uni => uni.mean))

  override def variance: DenseVector[Double] = DenseVector(marginalDistributions.map(uni => uni.variance))

  override def entropy: Double = marginalDistributions.map(uni => uni.entropy).sum

  override def mode: DenseVector[Double] = DenseVector(marginalDistributions.map(uni => uni.mode))

  override def draw(): DenseVector[Double] = DenseVector(marginalDistributions.map(uni => uni.draw()))
}
