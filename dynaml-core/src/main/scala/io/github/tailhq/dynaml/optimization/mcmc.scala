package io.github.tailhq.dynaml.optimization

import java.io.{File, PrintWriter}
import breeze.numerics.log
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.tailhq.dynaml.models.lm.KFilter
import io.github.tailhq.dynaml.models.lm.generalDLM._
import KFilter._
import scalaz.Scalaz._

object mcmc {
  case class MetropolisState(params: Parameters, accepted: Int, ll: Loglikelihood)

    /**
      * Moves the parameters according to a random walk
      */
  def perturb(delta: Double): Parameters => Parameters = {
    p =>
    Parameters(p.a + Gaussian(0, delta).draw, p.b + Gaussian(0, delta).draw, p.l1, p.l2)
  }
  /**
    * A metropolis hastings step
    */
  def metropolisStep(
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters)(
    state: MetropolisState): Option[(MetropolisState, MetropolisState)] = {

    val propParam = perturb(state.params)
    val propLl = likelihood(propParam)

    if (log(Uniform(0,1).draw) < propLl - state.ll) {
      Some((state, MetropolisState(propParam, state.accepted + 1, propLl)))
    } else {
      Some((state, state))
    }
  }

  /**
    * Generate iterations using scalaz unfold
    */
  def metropolisIters(
    initParams: Parameters,
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters): Stream[MetropolisState] = {

    val initState = MetropolisState(initParams, 0, likelihood(initParams))
    unfold(initState)(metropolisStep(likelihood, perturb))
  }

  def main(args: Array[String]): Unit = {
    val n = 10000
    val p = Parameters(3.0, 0.5, 0.0, 10.0)
    val observations = simulate(p).take(100).toVector

    val iters = metropolisIters(p, filterll(observations), perturb(0.1)).take(n)
    println(s"Accepted: ${iters.last.accepted.toDouble/n}")

    // write the parameters to file
    val pw = new PrintWriter(new File("data/mcmcOutRes.csv"))
    pw.write(iters.map(_.params).mkString("\n"))
    pw.close()
  }
}
