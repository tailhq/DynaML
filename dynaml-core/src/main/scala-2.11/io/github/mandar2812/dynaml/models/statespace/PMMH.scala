package io.github.mandar2812.dynaml.models.statespace

import breeze.stats.distributions.{Uniform, Rand, MultivariateGaussian, Process, MarkovChain}
import breeze.stats.distributions.Rand._
import breeze.stats.distributions.MarkovChain._
import breeze.linalg.DenseMatrix
import POMP._
import akka.stream.scaladsl._
import Stream._

case class MetropState(ll: LogLikelihood, params: Parameters, accepted: Int)

trait MetropolisHastings {

  /**
    * Proposal density, to propose new parameters for a model
    */
  def proposal: Parameters => Rand[Parameters]

  /**
    * Definition of the log-transition, used when calculating the acceptance ratio
    * This is the probability of moving between parameters according to the proposal distribution
    * Note: When using a symmetric proposal distribution (eg. Normal) this cancels in the acceptance ratio
    * @param from the previous parameter value
    * @param to the proposed parameter value
    */
  def logTransition(from: Parameters, to: Parameters): LogLikelihood

  val initialParams: Parameters

  /**
    * The likelihood function of the model, typically a pseudo-marginal likelihood for 
    * the PMMH algorithm
    */
  def logLikelihood: Parameters => LogLikelihood

  /**
    * Metropolis-Hastings step, for use in the iters function, to return an Akka stream of iterations
    */
  def mhStep: MetropState => Option[(MetropState, MetropState)] = p => {
    val propParams = proposal(p.params).draw
    val propll = logLikelihood(propParams)
    val a = propll - p.ll + logTransition(propParams, p.params) - logTransition(p.params, propParams)

    if (math.log(Uniform(0, 1).draw) < a) {
      Some((MetropState(propll, propParams, p.accepted + 1), p))
    } else {
      Some((p, p))
    }
  }

  /**
    * Generates an akka stream of MetropState, containing the current parameters, 
    * count of accepted moves and the current pseudo marginal log-likelihood
    */
  def iters: Source[MetropState, Any] = {
    val initState = MetropState(logLikelihood(initialParams), initialParams, 0)
    Source.unfold(initState)(mhStep)
  }

  /**
    * Return an akka stream of the parameters
    */
  def params: Source[Parameters, Any] = {
    iters map (_.params)
  }

  /**
    * A single step of the metropolis hastings algorithm to be used with breeze implementation of Markov Chain.
    * This is a slight alteration to the implementation in breeze, here MetropState holds on to the previous 
    * calculated pseudo marginal log-likelihood value so we don't need to re-run the particle filter each iteration
    */
  def mhStepRand: MetropState => Rand[MetropState] = p => {
    for {
      propParams <- proposal(p.params)
      propll = logLikelihood(propParams)
      a = propll - p.ll + logTransition(propParams, p.params) - logTransition(p.params, propParams)
      prop = if (math.log(Uniform(0,1).draw) < a) {
        MetropState(propll, propParams, p.accepted + 1)
      } else {
        p
      }
    } yield prop
  }

  /**
    * Use the Breeze Markov Chain to generate a process of MetropState
    * Process can be advanced by calling step
    */
  def breezeIters: Process[MetropState] = {
    val initState = MetropState(logLikelihood(initialParams), initialParams, 0)
    MarkovChain(initState)(mhStepRand)
  }
}

case class ParticleMetropolis(
  mll: Parameters => LogLikelihood,
  initParams: Parameters,
  perturb: Parameters => Rand[Parameters]) extends MetropolisHastings{

  def logLikelihood: Parameters => LogLikelihood = mll
  def logTransition(from: Parameters, to: Parameters): LogLikelihood = 0.0
  def proposal: Parameters => Rand[Parameters] = perturb
  val initialParams = initParams
}

case class ParticleMetropolisHastings(
  mll: Parameters => LogLikelihood,
  transitionProb: (Parameters, Parameters) => LogLikelihood,
  propParams: Parameters => Rand[Parameters],
  initParams: Parameters) extends MetropolisHastings {

  def logLikelihood: Parameters => LogLikelihood = mll
  def logTransition(from: Parameters, to: Parameters): LogLikelihood = transitionProb(from, to)
  def proposal: Parameters => Rand[Parameters] = propParams
  val initialParams = initParams
}
