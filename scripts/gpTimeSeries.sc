{
    import breeze.linalg.eig
    import breeze.stats.distributions.{ContinuousDistr, Gamma}
    import io.github.mandar2812.dynaml.kernels._
    import io.github.mandar2812.dynaml.models.bayes.{
      LinearTrendESGPrior,
      LinearTrendGaussianPrior,
      GaussianProcessPrior
    }
    import io.github.mandar2812.dynaml.modelpipe._
    import io.github.mandar2812.dynaml.probability._
    import io.github.mandar2812.dynaml.DynaMLPipe._
    import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
    import io.github.mandar2812.dynaml.analysis.implicits._
    import io.github.mandar2812.dynaml.optimization.GPMixtureMachine
    import io.github.mandar2812.dynaml.pipes.Encoder
    import io.github.mandar2812.dynaml.probability.distributions.UnivariateGaussian
    import spire.implicits._
  
    val rbfc      = new SECovFunc(0.5, 2.0)
    val mlpKernel = new MLP1dKernel(1.0, 1.0)
    val pKernel   = new Polynomial1dKernel(1, 1d) 
    val fbm       = new FBMCovFunction(0.5)
    val stKernel  = new TStudentCovFunc(0.1)
  
    val encoder = Encoder(
      (conf: Map[String, Double]) => (conf("c"), conf("s")),
      (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2)
    )
  
    val trendEncoder = Encoder(
      (cs: (Double, Double)) => Map("slope" -> cs._1, "intercept" -> cs._2),
      (conf: Map[String, Double]) => (conf("slope"), conf("intercept"))
    )

    val qtrendEncoder = Encoder(
      (cs: (Double, Double, Double)) => Map("a" -> cs._1, "b" -> cs._2, "d" -> cs._3),
      (conf: Map[String, Double]) => (conf("a"), conf("b"), conf("d"))
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
  
    val n = new MAKernel(1.0)
    
    val linear_trend = (-0.75, 0.2d)
    val quadratic_trend = (0d, 0.25, 0d)

    val brownian_motion = GaussianProcessPrior[Double, (Double, Double)](
      fbm,
      new MAKernel(0.001),
      MetaPipe((p: (Double, Double)) => (x: Double) => p._1 * x + p._2),
      trendEncoder,
      (0d, 0d)
    )
    brownian_motion.hyperPrior_(hyp_prior)

    val gp_prior = GaussianProcessPrior[Double, (Double, Double, Double)](
      mlpKernel + stKernel,
      n,
      MetaPipe((p: (Double, Double, Double)) => (x: Double) => p._1*x*x + p._2*x + p._3),
      qtrendEncoder,
      quadratic_trend
    )
    gp_prior.hyperPrior_(hyp_prior)

    val gpModelPipe = GPRegressionPipe[Seq[(Double, Double)], Double](
      identityPipe[Seq[(Double, Double)]],
      mlpKernel + stKernel,
      n,
      DataPipe((x:Double) => quadratic_trend._1*x*x + quadratic_trend._2*x + quadratic_trend._3)
    )

  
    val xs = Seq.tabulate[Double](50)(1d * _)
  
    val ys: MultGaussianPRV = gp_prior.priorDistribution(xs)
    val yb: MultGaussianPRV = brownian_motion.priorDistribution(xs)
  
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

    val samples =
      (1 to 10).map(_ => ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)
    val samples_br =
      (1 to 10).map(_ => yb.sample()).map(s => s.toBreezeVector.toArray.toSeq)
    val samples_ar = (1 to 10).map(_ => ys_ar.sample())

    val ys_ar_rec = RandomVariable[Seq[Double]](
      () => {
        // Generate y0 and y1
        val u0 = y0.draw 
        val u1 = gp_prior.priorDistribution(Seq(u0)).draw.toStream.head
        val d = Seq((u0, u1))
        val (_, xsamples) = (1 to xs.length - 2).scanLeft((d, u1))((dc, _) => {
          val (data, x) = dc
          val gpModel = gpModelPipe(data)
          val y = gpModel.predictiveDistribution(Seq(x)).draw.toStream.head
          (data :+ (x, y), y)
        }).unzip

        Seq(u0, u1) ++ xsamples
      })

    val samples_ar2 = (1 to 10).map(_ => ys_ar_rec.sample())

    

    spline(xs, samples.head)
    hold()
    samples.tail.foreach((s: Seq[Double]) => spline(xs, s))
    unhold()
    title("GP Explicit Time")
  
    spline(xs, samples_ar.head)
    hold()
    samples_ar.tail.foreach((s: Seq[Double]) => spline(xs, s))
    unhold()
    title("GP-AR")

    spline(xs, samples_ar2.head)
    hold()
    samples_ar2.tail.foreach((s: Seq[Double]) => spline(xs, s))
    unhold()
    title("GP-AR Recurrent")

    spline(xs, samples_br.head)
    hold()
    samples_br.tail.foreach((s: Seq[Double]) => spline(xs, s))
    unhold()
    title("Brownian Motion")

    /* val phase_space_traj = samples_ar2.map(xs => xs.sliding(2).toSeq.map(h => (h.head, h.last)))
    
    line(phase_space_traj.head.map(_._1), phase_space_traj.head.map(_._2))
    hold()
    phase_space_traj.tail.foreach((s: Seq[(Double, Double)]) => line(s.map(_._1), s.map(_._2)))
    unhold()
    title("Gaussian Process AR Phase Space Trajectories") */

}