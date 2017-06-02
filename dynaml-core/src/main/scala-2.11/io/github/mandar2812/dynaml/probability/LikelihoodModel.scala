package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * @author mandar2812 date: 07/01/2017.
  * A probability distribution driven by a set
  * of parameters.
  */
trait LikelihoodModel[Q] extends DataPipe[Q, Double]{


  /**
    * Returns log p(q)
    * */
  def loglikelihood(q: Q): Double

  override def run(data: Q) = loglikelihood(data)

}

object LikelihoodModel {
  def apply[Q, T <: Density[Q] with Rand[Q]](d: T) = new LikelihoodModel[Q] {
    val density = d

    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = density.logApply(q)

  }

  def apply[Q](d: ContinuousDistr[Q]) = new LikelihoodModel[Q] {
    val density = d

    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = density.logPdf(q)
  }

  def apply[Q](f: (Q) => Double) = new LikelihoodModel[Q] {
    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = f(q)
  }

}

trait DifferentiableLikelihoodModel[Q] extends LikelihoodModel[Q] {
  /**
    * Returns d(log p(q|r))/dr
    * */
  def gradLogLikelihood(q: Q): Q
}

object DifferentiableLikelihoodModel {

  def apply[Q](d: ContinuousDistr[Q], grad: DataPipe[Q, Q]): DifferentiableLikelihoodModel[Q] =
    new DifferentiableLikelihoodModel[Q] {

      val density = d

      val gradPotential = grad

      /**
        * Returns log p(q)
        **/
      override def loglikelihood(q: Q) = density.logPdf(q)

      /**
        * Returns d(log p(q|r))/dr
        **/
      override def gradLogLikelihood(q: Q) = gradPotential(q)
    }

  def apply[Q](f: (Q) => Double, grad: (Q) => Q) = new DifferentiableLikelihoodModel[Q] {
    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = f(q)

    /**
      * Returns d(log p(q|r))/dr
      **/
    override def gradLogLikelihood(q: Q) = grad(q)
  }
}