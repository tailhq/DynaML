/**
  * Created by mandar on 28/7/16.
  */

import breeze.stats.distributions._
import io.github.mandar2812.dynaml.probability.{IIDRandomVarDistr, ProbabilityModel, RandomVariable}
import spire.implicits._

val n = RandomVariable(Gaussian(0.0, 0.25))
val n1 = RandomVariable(Laplace(0.0, 0.25))

val combinedDist = n :*: n1

val cond2 = DataPipe((xy: (Double, Double)) => RandomVariable(new Gumbel(0.4*xy._1 + 0.6*xy._2, math.sin(xy._1+xy._2)/2.0)))

val mod = ProbabilityModel(combinedDist, cond2)

val combinedDist2 = n1 :*: n

val mod1 = ProbabilityModel(combinedDist2, cond2)


val cointoss = RandomVariable(new Binomial(1, 0.5))

val mixture = DataPipe((toss: Int) => if(toss == 1) mod else mod1)

val mixtureModel = ProbabilityModel(cointoss, mixture)


val p = RandomVariable(new Beta(0.5, 0.5))

val coinLikihood = DataPipe((p: Double) => new BinomialRV(500, p))

val c_model = ProbabilityModel(p, coinLikihood)

val post = c_model.posterior(375)

histogram((1 to 2000).map(_ => p.sample()))

histogram((1 to 2000).map(_ => post.sample()))

val priorMean = RandomVariable(new Gaussian(0.0, 1.0))

val priorSigma = RandomVariable(new Gamma(7.5, 1.0))

val prior = priorSigma :*: priorMean

val iidPrior = IIDRandomVarDistr(prior) _

scatter(iidPrior(1000).sample())

val likelihood = DataPipe((params: (Double, Double)) =>
  IIDRandomVarDistr(RandomVariable(new Gaussian(params._1, params._2)))(2000))

val gModel = ProbabilityModel(prior, likelihood)

val posterior = gModel.posterior(new Gaussian(-1.0, 2.0).sample(2000).toStream)

scatter((1 to 2000).map(_ => posterior.sample()))
