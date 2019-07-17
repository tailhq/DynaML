{
  import breeze.linalg.eig
  import breeze.stats.distributions.{ContinuousDistr, Gamma}
  import io.github.mandar2812.dynaml.kernels._
  import io.github.mandar2812.dynaml.models.bayes.{
    LinearTrendESGPrior,
    LinearTrendGaussianPrior,
    GaussianProcessPrior
  }
  import io.github.mandar2812.dynaml.probability._
  import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
  import io.github.mandar2812.dynaml.analysis.implicits._
  import io.github.mandar2812.dynaml.optimization.GPMixtureMachine
  import io.github.mandar2812.dynaml.pipes.Encoder
  import io.github.mandar2812.dynaml.probability.distributions.UnivariateGaussian
  import spire.implicits._

  val rbfc      = new SECovFunc(0.5, 1.0)
  val mlpKernel = new MLP1dKernel(0.25, 0.25)

  val encoder = Encoder(
    (conf: Map[String, Double]) => (conf("c"), conf("s")),
    (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2)
  )

  val trendEncoder = Encoder(
    (cs: (Double, Double)) => Map("slope" -> cs._1, "intercept" -> cs._2),
    (conf: Map[String, Double]) => (conf("slope"), conf("intercept"))
  )

  val hyp_prior
    : Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] = Map(
    "c"          -> RandomVariable(UnivariateGaussian(2.5, 1.5)),
    "s"          -> RandomVariable(Gamma(2.0, 2.0)),
    "noiseLevel" -> RandomVariable(Gamma(2.0, 2.0))
  )

  val sgp_hyp_prior = hyp_prior ++ Map(
    "cutoff"   -> RandomVariable(UnivariateGaussian(0.0, 1.0)),
    "skewness" -> RandomVariable(UnivariateGaussian(0.0, 1.0))
  )

  val gsmKernel       = GaussianSpectralKernel[Double](3.5, 2.0, encoder)
  val cubsplineKernel = new CubicSplineKernel[Double](2.0)

  val n = new MAKernel(0.8)

  val gp_prior = GaussianProcessPrior[Double, (Double, Double)](
    mlpKernel,
    n,
    MetaPipe((p: (Double, Double)) => (x: Double) => p._1 * x + p._2),
    trendEncoder,
    (0.5d, -0.25d)
  )
  gp_prior.hyperPrior_(hyp_prior)

  val sgp_prior =
    new LinearTrendESGPrior[Double](rbfc, n, trendEncoder, 0.75, 0.1, 0.0, 0.0)
  sgp_prior.hyperPrior_(sgp_hyp_prior)

  val xs = Seq.tabulate[Double](20)(1d * _)

  val ys: MultGaussianPRV = gp_prior.priorDistribution(xs)

  val y0 = RandomVariable(UnivariateGaussian(0.0, 1.0))
  val ys_ar: RandomVariable[Seq[Double]] =
    RandomVariable(
      () =>
        xs.map(_.toInt)
          .scanLeft(y0.draw)(
            (u: Double, _: Int) =>
              gp_prior.priorDistribution(Seq(u)).draw.toStream.head
          )
    )

  val sgp_ys: BlockedMESNRV = sgp_prior.priorDistribution(xs)

  val samples =
    (1 to 8).map(_ => ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)
  val samples_ar = (1 to 8).map(_ => ys_ar.sample())
  val samples_sgp =
    (1 to 8).map(_ => sgp_ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)

  val gammaRV  = RandomVariable(new Gamma(2.0, 2.0))
  val noiseAdd = GaussianRV(0.0, 0.2)

  val dataset = xs.map { i =>
    (
      i + GaussianRV(0.0, 0.02).sample(),
      noiseAdd.sample() + math.cos(i) * math.exp(-0.25 * i)
    )
  }

  //Set hyper-parameter selection configuration
  gp_prior.globalOptConfig_(
    Map(
      "gridStep"  -> "0.15",
      "gridSize"  -> "5",
      "globalOpt" -> "GPC",
      "policy"    -> "GS"
    )
  )
  sgp_prior.globalOptConfig_(Map("gridStep" -> "0.15", "gridSize" -> "4"))

  val gpModel  = gp_prior.posteriorModel(dataset)
  val sgpModel = sgp_prior.posteriorModel(dataset)

  gp_prior.globalOptConfig_(
    Map(
      "gridStep"  -> "0.0",
      "gridSize"  -> "1",
      "globalOpt" -> "GS",
      "policy"    -> "GS"
    )
  )
  val gpModel1 = gp_prior.posteriorModel(dataset)

  val mixt_machine = new GPMixtureMachine(gpModel1)
    .setPrior(hyp_prior)
    .setGridSize(2)
    .setStepSize(0.50)
    .setLogScale(true)
    .setMaxIterations(200)
    .setNumSamples(3)

  val (mix_model, mixt_model_conf) =
    mixt_machine.optimize(
      gp_prior.covariance.effective_state ++ gp_prior.noiseCovariance.effective_state
    )

  val zs: MultGaussianPRV   = gpModel.predictiveDistribution(xs)
  val sgp_zs: BlockedMESNRV = sgpModel.predictiveDistribution(xs)
  val mix_zs                = mix_model.predictiveDistribution(xs)

  val MultGaussianPRV(m, c) = zs
  val eigD                  = eig(c.toBreezeMatrix)
  val eValuesPositive       = eigD.eigenvalues.toArray.forall(_ >= 0.0)

  if (eValuesPositive) {
    val samplesPost =
      zs.iid(8).sample().map(s => s.toBreezeVector.toArray.toSeq)

    spline(xs, samplesPost.head)
    hold()
    samplesPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
    unhold()
    title("Gaussian Process posterior samples")

  } else {
    println("Predictive Covariance Ill-Posed!")
  }

  val (dx, dy) = dataset.sorted.unzip

  val samplesSGPPost =
    sgp_zs.iid(8).sample().map(_.toBreezeVector.toArray.toSeq)

  val samplesMixPost =
    mix_zs.iid(8).sample().map(_.toBreezeVector.toArray.toSeq)

  spline(xs, samplesSGPPost.head)
  hold()
  samplesSGPPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Ext. Skew Gaussian Process posterior samples")

  spline(xs, samplesMixPost.head)
  hold()
  samplesMixPost.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Gaussian Process Mixture posterior samples")

  spline(dx, dy)
  title("Data")

  spline(xs, samples.head)
  hold()
  samples.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Gaussian Process prior samples")

  spline(xs, samples_ar.head)
  hold()
  samples_ar.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Gaussian Process AR prior samples")

  spline(xs, samples_sgp.head)
  hold()
  samples_sgp.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title("Ext. Skew Gaussian Process prior samples")

}
