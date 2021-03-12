package io.github.tailhq.dynaml.models.stp

import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.tailhq.dynaml.analysis.InnerProductPV
import io.github.tailhq.dynaml.models.GenContinuousMixtureModel
import io.github.tailhq.dynaml.probability.MultStudentsTPRV
import io.github.tailhq.dynaml.probability.distributions.BlockedMultivariateStudentsT
import spire.algebra.VectorSpace

import scala.reflect.ClassTag

/**
  * @author tailhq date 21/06/2017.
  * */
class StudentTProcessMixture[T, I: ClassTag](
  override val component_processes: Seq[AbstractSTPRegressionModel[T, I]],
  override val weights: DenseVector[Double]) extends
  GenContinuousMixtureModel[
    T, I, Double, PartitionedVector,
    PartitionedPSDMatrix, BlockedMultivariateStudentsT,
    MultStudentsTPRV, AbstractSTPRegressionModel[T, I]](
    component_processes, weights) {

  protected val blockSize: Int = component_processes.head._blockSize

  override protected def toStream(y: PartitionedVector): Stream[Double] = y.toStream

  override protected def getVectorSpace(num_dim: Int): VectorSpace[PartitionedVector, Double] =
    InnerProductPV(num_dim, blockSize)
}
