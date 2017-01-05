import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PushforwardMap}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.{GaussianRV, RandomVariable}
import spire.implicits._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.probability.distributions.SkewGaussian
/**
  * @author mandar date 22/12/2016.
  */

val g = GaussianRV(0.0, 0.25)

val sg = RandomVariable(SkewGaussian(1.0, 0.0, 0.25))

implicit val detImpl = identityPipe[Double]

val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.exp(x)),
  DifferentiableMap(
    (x: Double) => math.log(x),
    (x: Double) => 1.0/x)
)

val p = h->g

val q = h->sg

val y = Array.tabulate[(Double, Double)](100)(n => (n*0.03, q.underlyingDist.pdf(n*0.03)))
spline(y.toIterable)
hold()
val x = Array.tabulate[(Double, Double)](100)(n => (n*0.03, p.underlyingDist.pdf(n*0.03)))
spline(x.toIterable)
unhold()
legend(List("Log Skew Gaussian", "Log Gaussian"))
title("Probability Density Functions")
