import io.github.mandar2812.dynaml.kernels.{MAKernel, RBFCovFunc}
import io.github.mandar2812.dynaml.models.bayes.LinearTrendGaussianPrior
import io.github.mandar2812.dynaml.probability.MultGaussianPRV
import io.github.mandar2812.dynaml.analysis.implicits._
import com.quantifind.charts.Highcharts._


val rbfc = new RBFCovFunc(1.0)
val n = new MAKernel(0.1)

val gp_prior = new LinearTrendGaussianPrior[Double](rbfc, n, 0.5)

val xs = Seq.tabulate[Double](20)(0.5*_)

val ys: MultGaussianPRV = gp_prior.priorDistribution(xs)

val samples = (1 to 8).map(_ => ys.sample()).map(s => s.toBreezeVector.toArray.toSeq)

spline(xs, samples.head)
hold()
samples.tail.foreach((s: Seq[Double]) => spline(xs, s))
unhold()
