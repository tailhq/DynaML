/**
  * Created by mandar on 28/7/16.
  */

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import breeze.stats.mcmc.ThreadedBufferedRand
import io.github.mandar2812.dynaml.kernels.PeriodicKernel
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability._
import spire.implicits._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.utils

val data = new Gaussian(-2.0, 3.25).sample(2000).toStream

val priorMean = RandomVariable(new Gaussian(0.0, 2.5))

val priorSigma = RandomVariable(new Gamma(2.0, 2.0))

val prior = priorMean :* priorSigma

val iidPrior = IIDRandomVarDistr(prior) _

val likelihood = DataPipe((params: (Double, Double)) =>
  IIDRandomVarDistr(RandomVariable(new Gaussian(params._1, params._2)))(2000))

implicit val pField = utils.productField[Double, Double]

val prop =
  RandomVariable(new Gaussian(0.0, 0.4)) :* RandomVariable(new Gaussian(0.0, 0.1))

val gModel = new ContinuousMCMCModel(prior, likelihood, prop, 10000L)

val posterior = gModel.posterior(data)

val samples = (1 to 3000).map(_ => {posterior.sample()})

histogram(data)
title("Histogram of data")


scatter(iidPrior(3000).sample())
hold()
scatter(samples)
legend(List("Prior", "Posterior"))
xAxis("Mean")
yAxis("Std Deviation")
unhold()



val p = RandomVariable(new Beta(7.5, 7.5))

val coinLikelihood = DataPipe((p: Double) => BinomialRV(500, p))

val c_model = ProbabilityModel(p, coinLikelihood)

val post = c_model.posterior(350)

histogram((1 to 2000).map(_ => p.sample()))
hold()
histogram((1 to 2000).map(_ => post.sample()))

unhold()
