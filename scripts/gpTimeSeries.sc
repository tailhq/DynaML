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

  val p = 4
  val num_sample_paths = 10
  val len = 100

  val xs = Seq.tabulate[Double](len)(1d * _)

  implicit val ev = VectorField(p)

  val y0 = RandomVariable(UnivariateGaussian(0.0, 1.0))

  val urv = UniformRV(-0.75, 0.25)

  val coeff_gen = (order: Int) => DenseVector.tabulate(order)(_ => urv.draw)

  val w = coeff_gen(p)

  val rbfc      = new SEKernel(1d, 2.0)
  val mlpKernel = new MLPKernel(0.5, 0.5d)
  val fbm       = new FBMCovFunction(0.5)
  val stKernel  = new TStudentKernel(0.2)
  val perKernel = new PeriodicCovFunc(2d, 1.5d, 0.1d)
  val arpKernel = new GenericMaternKernel[Double](2.5, p)
  val n         = new MAKernel(1.0)
  val noise     = new DiracKernel(0.5d)

  val linear_coeff    = (0d, 0d)
  val quadratic_coeff = (-0.01d, -0.75d, 0d)

  //Define the trend functions. One a linear trend
  //and the other a parabola.

  val linear_vec_trend = MetaPipe(
    (p: DenseVector[Double]) => (x: DenseVector[Double]) => p dot x
  )

  val linear_trend_mean = MetaPipe(
    (p: (Double, Double)) =>
      (x: Double) => {
        (p._1 * x) + p._2
      }
  )

  val quadratic_trend_mean = MetaPipe(
    (p: (Double, Double, Double)) =>
      (x: Double) => {
        (p._1 * x * x) + (p._2 * x) + p._3
      }
  )

  //Determine how the trend coefficients map to key-value pairs
  val linear_trend_encoder = Encoder(
    (cs: (Double, Double)) => Map("slope" -> cs._1, "intercept" -> cs._2),
    (conf: Map[String, Double]) => (conf("slope"), conf("intercept"))
  )

  val linear_vec_trend_encoder = Encoder(
    (cs: DenseVector[Double]) => {
      cs.toArray.zipWithIndex.map(cp => (s"w_${cp._2}", cp._1)).toMap
    },
    (conf: Map[String, Double]) =>
      DenseVector(conf.toSeq.sortBy(_._1).map(_._2).toArray)
  )

  val quadratic_trend_encoder = Encoder(
    (cs: (Double, Double, Double)) =>
      Map("a" -> cs._1, "b" -> cs._2, "d" -> cs._3),
    (conf: Map[String, Double]) => (conf("a"), conf("b"), conf("d"))
  )

  //Define a RBF covariance based gaussian process
  val gp_explicit = GaussianProcessPrior[Double, (Double, Double)](
    arpKernel,
    new MAKernel(0.5d),
    linear_trend_mean,
    linear_trend_encoder,
    linear_coeff
  )

  //Define a gaussian process for GP Time Series models.
  val gp_prior = GaussianProcessPrior[DenseVector[Double], DenseVector[Double]](
    (mlpKernel + stKernel)*0.5,
    noise,
    linear_vec_trend,
    linear_vec_trend_encoder,
    w
  )

  val gpModelPipe = 
    new GPBasisFuncRegressionPipe[Seq[(DenseVector[Double], Double)], DenseVector[Double]](
      identityPipe[Seq[(DenseVector[Double], Double)]],
      rbfc,//(mlpKernel + stKernel)*0.5,
      noise,
      identityPipe[DenseVector[Double]],
      MultGaussianRV(
        DenseVector.zeros[Double](p), 
        diag(DenseVector.fill[Double](p)(0.01d))
      )
    )
    /* GPRegressionPipe[Seq[(DenseVector[Double], Double)], DenseVector[Double]](
      identityPipe[Seq[(DenseVector[Double], Double)]],
      mlpKernel + stKernel,
      noise,
      linear_vec_trend(w)
    ) */

  

  val y_explicit: MultGaussianPRV = gp_explicit.priorDistribution(xs)

  val samples_gp_explicit =
    y_explicit.iid(10).draw.map(s => s.toBreezeVector.toArray.toSeq).toSeq

  val ys_ar_rec = RandomVariable[Seq[Double]](() => {
    // Generate y0 and y1
    val u0: DenseVector[Double] = DenseVector.tabulate(p)(_ => y0.draw)
    val u1: Double              = gp_prior.priorDistribution(Seq(u0)).draw.toStream.head
    val d                       = Seq((u0, u1))
    val (_, xsamples) =
      (1 to xs.length - p - 1)
        .scanLeft((d, DenseVector(Array(u1) ++ u0(0 to -2).toArray)))(
          (dc, _) => {

            val (data, x) = dc
            val gpModel   = gpModelPipe(data)
            val y         = gpModel.predictiveDistribution(Seq(x)).draw.toStream.head
            (data :+ (x, y), DenseVector(Array(y) ++ x(0 to -2).toArray))
          }
        )
        .unzip

    u0.toArray.toSeq ++ xsamples.map(_(0))
  })

  val markov_process = (n: Int) =>
    RandomVariable[Seq[Double]](() => {
      val u0 = DenseVector.tabulate(p)(_ => y0.draw)
      val xs_tail = (1 to n - p - 1)
        .scanLeft(
          u0
        )((y: DenseVector[Double], _) => {
          val y_new: Double = linear_vec_trend(w)(y) + y0.draw
          DenseVector(Array(y_new) ++ y(0 to -2).toArray)
        })
        .toSeq
        .map(_(0))

      u0.toArray.toSeq ++ xs_tail
    })

  val samples_ar_rec = ys_ar_rec.iid(10).draw.toSeq

  val samples_markov = markov_process(xs.length).iid(10).draw.toSeq

  spline(xs, samples_gp_explicit.head)
  hold()
  samples_gp_explicit.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title(
    s"""GP Explicit Time AR($p): ${gp_explicit.covariance.toString
      .split("\\.")
      .last}"""
  )

  val markov_formula = w.toArray.toSeq.zipWithIndex
    .map(
      cp =>
        if (cp._1 >= 0d && cp._2 == 0) f"${cp._1}%3.2f*y(t-${cp._2 + 1})"
        else if (cp._1 >= 0d) f"+ ${cp._1}%3.2f*y(t-${cp._2 + 1})"
        else f" ${cp._1}%3.2f*y(t-${cp._2 + 1})"
    )
    .reduceLeft(_ ++ _)

  spline(xs, samples_markov.head)
  hold()
  samples_markov.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title(
    s"""AR($p): y(t) = ${markov_formula} + noise"""
  )

  spline(xs, samples_ar_rec.head)
  hold()
  samples_ar_rec.tail.foreach((s: Seq[Double]) => spline(xs, s))
  unhold()
  title(
    s"""GP-AR Recurrent: ${gpModelPipe.covariance.toString.split("\\.").last}"""
  )

  println(s"Linear coefficients: ${w.toArray.toSeq}")

}
