import breeze.linalg.eig
import breeze.stats.distributions.{ContinuousDistr, Gamma, Gaussian}
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.bayes.{LinearTrendESGPrior, LinearTrendGaussianPrior}
import io.github.mandar2812.dynaml.probability._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.optimization.ProbGPMixtureMachine
import io.github.mandar2812.dynaml.pipes.Encoder
import io.github.mandar2812.dynaml.probability.distributions.UnivariateGaussian

val rbfc = new RBFCovFunc(1.5)

val encoder = Encoder(
  (conf: Map[String, Double]) => (conf("c"), conf("s")),
  (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2))

val trendEncoder = Encoder(
  (cs: (Double, Double)) => Map("slope" -> cs._1, "intercept" -> cs._2),
  (conf: Map[String, Double]) => (conf("slope"), conf("intercept"))
)

val hyp_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] = Map(
  "c" -> RandomVariable(UnivariateGaussian(2.5, 1.5)),
  "s" -> RandomVariable(Gamma(2.0, 2.0)),
  "noiseLevel" -> RandomVariable(Gamma(2.0, 2.0)))

val sgp_hyp_prior = hyp_prior ++ Map(
  "cutoff" -> RandomVariable(UnivariateGaussian(0.0, 1.0)),
  "skewness" -> RandomVariable(UnivariateGaussian(0.0, 1.0)))

val gsmKernel = GaussianSpectralKernel[Double](3.5, 2.0, encoder)
val n = new MAKernel(0.8)

val gp_prior = new LinearTrendGaussianPrior[Double](gsmKernel, n, trendEncoder, 0.0, 0.0)
//gp_prior.hyperPrior_(hyp_prior)

val sgp_prior = new LinearTrendESGPrior[Double](rbfc, n, trendEncoder, 0.75, 0.1, 0.0, 0.0)
sgp_prior.hyperPrior_(sgp_hyp_prior)

val xs = Seq.tabulate[Double](20)(0.5*_)

val ys: MultGaussianPRV = gp_prior.priorDistribution(xs)
val sgp_ys: BlockedMESNRV = sgp_prior.priorDistribution(xs)

val samples = (1 to 8).map(_ => ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)

val samples_sgp = (1 to 8).map(_ => sgp_ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)

val gammaRV = RandomVariable(new Gamma(2.0, 2.0))
val noiseAdd = GaussianRV(0.0, 0.2)

val dataset = xs.map{i => (i + GaussianRV(0.0, 0.02).sample(), noiseAdd.sample()+math.cos(i)*math.exp(-0.25*i))}

//Set hyper-parameter selection configuration
gp_prior.globalOptConfig_(Map("gridStep" -> "0.15", "gridSize" -> "5", "globalOpt" -> "GPC", "policy" -> "GS"))
sgp_prior.globalOptConfig_(Map("gridStep" -> "0.15", "gridSize" -> "4"))

val gpModel = gp_prior.posteriorModel(dataset)
val sgpModel = sgp_prior.posteriorModel(dataset)


gp_prior.globalOptConfig_(Map("gridStep" -> "0.0", "gridSize" -> "1", "globalOpt" -> "GS", "policy" -> "GS"))
val gpModel1 = gp_prior.posteriorModel(dataset)

val mixt_machine = new ProbGPMixtureMachine(gpModel1)
  .setGridSize(2)
  .setStepSize(0.50)
  .setLogScale(true)
  .setMaxIterations(16)

val (mix_model, mixt_model_conf) = mixt_machine.optimize(gp_prior.covariance.state ++ gp_prior.noiseCovariance.state)


val zs: MultGaussianPRV = gpModel.predictiveDistribution(xs)
val sgp_zs: BlockedMESNRV = sgpModel.predictiveDistribution(xs)
val mix_zs = mix_model.predictiveDistribution(xs)

val MultGaussianPRV(m, c) = zs
val eigD = eig(c.toBreezeMatrix)
val eValuesPositive = eigD.eigenvalues.toArray.forall(_ >= 0.0)

if(eValuesPositive) {
  val samplesPost = zs.iid(8).sample().map(s => s.toBreezeVector.toArray.toSeq)

  spline(xs, samplesPost.head)
  hold()
  samplesPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Gaussian Process posterior samples")

} else {
  println("Predictive Covariance Ill-Posed!")
}

val samplesSGPPost = sgp_zs.iid(8).sample().map(_.toBreezeVector.toArray.toSeq)

spline(xs, samplesSGPPost.head)
hold()
samplesSGPPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
unhold()
title("Ext. Skew Gaussian Process posterior samples")

val samplesMixPost = mix_zs.iid(8).sample().map(_.toBreezeVector.toArray.toSeq)

spline(xs, samplesMixPost.head)
hold()
samplesMixPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
unhold()
title("Gaussian Process Mixture posterior samples")


val (dx, dy) = dataset.sorted.unzip

spline(dx, dy)
title("Data")

spline(xs, samples.head)
hold()
samples.tail.foreach((s: Seq[Double]) => spline(xs, s))
unhold()
title("Gaussian Process prior samples")

spline(xs, samples_sgp.head)
hold()
samples_sgp.tail.foreach((s: Seq[Double]) => spline(xs, s))
unhold()
title("Ext. Skew Gaussian Process prior samples")
