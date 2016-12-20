/**
  * Created by mandar on 28/7/16.
  */

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.kernels.PeriodicKernel
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability._
import spire.implicits._
import com.quantifind.charts.Highcharts._


val p = RandomVariable(new Beta(7.5, 7.5))

val coinLikelihood = DataPipe((p: Double) => BinomialRV(500, p))

val c_model = ProbabilityModel(p, coinLikelihood)

val post = c_model.posterior(350)

histogram((1 to 2000).map(_ => p.sample()))
hold()
histogram((1 to 2000).map(_ => post.sample()))

unhold()

val data = new Gaussian(-2.0, 3.25).sample(2000).toStream
histogram(data)
title("Histogram of data")

val priorMean = RandomVariable(new Gaussian(0.5, 1.0))

val priorSigma = RandomVariable(new Gamma(2.0, 1.0))

val prior = priorMean :* priorSigma

val iidPrior = IIDRandomVarDistr(prior) _

scatter(iidPrior(1000).sample())
hold()
val likelihood = DataPipe((params: (Double, Double)) =>
  IIDRandomVarDistr(RandomVariable(new Gaussian(params._1, params._2)))(2000))

val gModel = ProbabilityModel(prior, likelihood)

val posterior = gModel.posterior(data)

val samples = (1 to 1000).map(i => {posterior.sample()})

scatter(samples)
legend(List("Prior", "Posterior"))
xAxis("Mean")
yAxis("Std Deviation")
unhold()
