package io.github.mandar2812.dynaml.analysis

import breeze.numerics.log
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}
import io.github.mandar2812.dynaml.probability.{ContinuousDistrRV, RandomVarWithDistr, RandomVariable}
import spire.algebra.Field

/**
  * @author mandar2812 on 22/12/2016.
  *
  * Push forward map is a function that has a well defined inverse
  * as well as Jacobian of the inverse.
  */
abstract class PushforwardMap[
@specialized(Double) Source,
@specialized(Double) Destination,
@specialized(Double) Jacobian](
  implicit detImpl: DataPipe[Jacobian, Double], field: Field[Destination])
  extends Encoder[Source, Destination] { self =>
  /**
    * Represents the decoding/inverse operation.
    */
  override val i: DifferentiableMap[Destination, Source, Jacobian]

  def ->[R <: ContinuousDistrRV[Source]](r: R)
  : RandomVarWithDistr[Destination, ContinuousDistr[Destination]] =
    RandomVariable(new ContinuousDistr[Destination] {
      override def unnormalizedLogPdf(x: Destination) =
        r.underlyingDist.unnormalizedLogPdf(i(x)) + log(detImpl(i.J(x)))

      override def logNormalizer = r.underlyingDist.logNormalizer

      override def draw() = self.run(r.underlyingDist.draw())
    })

}

object PushforwardMap {
  def apply[S, D, J](forward: DataPipe[S, D], reverse: DifferentiableMap[D, S, J])(
    implicit detImpl: DataPipe[J, Double], field: Field[D]) =
    new PushforwardMap[S, D, J] {
      /**
        * Represents the decoding/inverse operation.
        */
      override val i = reverse

      override def run(data: S) = forward(data)
    }
}
