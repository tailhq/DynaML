package io.github.tailhq.dynaml.modelpipe

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Moments}
import io.github.tailhq.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.tailhq.dynaml.models.gp.AbstractGPRegressionModel
import io.github.tailhq.dynaml.models.stp.{AbstractSTPRegressionModel, MVStudentsTModel}
import io.github.tailhq.dynaml.models.{
ContinuousProcessModel, GenContinuousMixtureModel,
SecondOrderProcessModel, StochasticProcessMixtureModel}
import io.github.tailhq.dynaml.optimization.GloballyOptimizable
import io.github.tailhq.dynaml.pipes.DataPipe2
import io.github.tailhq.dynaml.probability.{ContinuousRVWithDistr, MatrixTRV, MultGaussianPRV, MultStudentsTPRV}
import io.github.tailhq.dynaml.probability.distributions.{
BlockedMultiVariateGaussian, BlockedMultivariateStudentsT,
HasErrorBars, MatrixT}

import scala.reflect.ClassTag

/**
  * Mixture Pipe takes a sequence of stochastic process models
  * and associated probability weights and returns a mixture model.
  * @author tailhq date 22/06/2017.
  * */
abstract class MixturePipe[
T, I: ClassTag, Y, YDomain, YDomainVar,
BaseDistr <: ContinuousDistr[YDomain]
  with Moments[YDomain, YDomainVar]
  with HasErrorBars[YDomain],
W1 <: ContinuousRVWithDistr[YDomain, BaseDistr],
BaseProcess <: ContinuousProcessModel[T, I, Y, W1]
  with SecondOrderProcessModel[T, I, Y, Double, DenseMatrix[Double], W1]
  with GloballyOptimizable] extends
  DataPipe2[Seq[BaseProcess], DenseVector[Double],
    GenContinuousMixtureModel[
      T, I, Y, YDomain, YDomainVar,
      BaseDistr, W1, BaseProcess]]


class GPMixturePipe[T, I: ClassTag] extends
  MixturePipe[T, I, Double, PartitionedVector, PartitionedPSDMatrix,
    BlockedMultiVariateGaussian, MultGaussianPRV,
    AbstractGPRegressionModel[T, I]] {

  override def run(
    models: Seq[AbstractGPRegressionModel[T, I]],
    weights: DenseVector[Double]) =
    StochasticProcessMixtureModel(models, weights)
}

class StudentTMixturePipe[T, I: ClassTag] extends
  MixturePipe[T, I, Double, PartitionedVector, PartitionedPSDMatrix,
    BlockedMultivariateStudentsT, MultStudentsTPRV,
    AbstractSTPRegressionModel[T, I]] {

  override def run(
    models: Seq[AbstractSTPRegressionModel[T, I]],
    weights: DenseVector[Double]) =
    StochasticProcessMixtureModel(models, weights)
}

class MVStudentsTMixturePipe[T, I: ClassTag] extends
  MixturePipe[
    T, I, DenseVector[Double], DenseMatrix[Double],
    (DenseMatrix[Double], DenseMatrix[Double]),
    MatrixT, MatrixTRV,
    MVStudentsTModel[T, I]] {

  override def run(
    models: Seq[MVStudentsTModel[T, I]],
    weights: DenseVector[Double]) =
    StochasticProcessMixtureModel(models, weights)
}
