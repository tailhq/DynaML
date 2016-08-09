/**
  * Created by mandar on 28/7/16.
  */

import breeze.stats.distributions._
import io.github.mandar2812.dynaml.probability.{IIDRandomVarDistr, ProbabilityModel, RandomVariable}
import spire.implicits._

val p = RandomVariable(new Beta(1.5, 0.75))

val coinLikihood = DataPipe((p: Double) => new BinomialRV(500, p))

val c_model = ProbabilityModel(p, coinLikihood)

val post = c_model.posterior(225)

histogram((1 to 2000).map(_ => p.sample()))
hold()
histogram((1 to 2000).map(_ => post.sample()))

val priorMean = RandomVariable(new Gaussian(0.0, 1.0))

val priorSigma = RandomVariable(new Gamma(7.5, 1.0))

val prior = priorMean :* priorSigma

val iidPrior = IIDRandomVarDistr(prior) _

scatter(iidPrior(1000).sample())
hold()
val likelihood = DataPipe((params: (Double, Double)) =>
  IIDRandomVarDistr(RandomVariable(new Gaussian(params._1, params._2)))(2000))

val gModel = ProbabilityModel(prior, likelihood)

val posterior = gModel.posterior(new Gaussian(-2.0, 0.25).sample(2000).toStream)

val samples = (1 to 1000).map(i => {posterior.sample()})

scatter(samples)
