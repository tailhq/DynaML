package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.probability.distributions.BlockedMESN

/**
  * Created by mandar on 28/02/2017.
  */
case class BlockedMESNRV(
  tau: Double, alpha: PartitionedVector,
  mu: PartitionedVector, sigma: PartitionedPSDMatrix)
  extends ContinuousDistrRV[PartitionedVector] {

  override val underlyingDist = BlockedMESN(tau, alpha, mu, sigma)

}
