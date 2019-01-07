import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PushforwardMap}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.{E, GaussianRV, RandomVariable, OrderStats}
import spire.implicits._
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import io.github.mandar2812.dynaml.probability.distributions.{SkewGaussian, UESN}

val g = GaussianRV(0.0, 0.5)
val sg = RandomVariable(SkewGaussian(10.0, 0.0, 0.5))

val uesg = RandomVariable(UESN(0.0, 0.45, 0.0, 0.005))

OrderStats(uesg, 0.5)
OrderStats(sg, 0.5)
OrderStats(g, 0.5)

val u = Array.tabulate[(Double, Double)](100)(n => (n*0.05 - 2.5, uesg.underlyingDist.pdf(n*0.05 - 2.5)))
spline(u.toIterable)
hold()
val v = Array.tabulate[(Double, Double)](100)(n => (n*0.05 - 2.5, g.underlyingDist.pdf(n*0.05 - 2.5)))
spline(v.toIterable)
val w = Array.tabulate[(Double, Double)](100)(n => (n*0.05 - 2.5, sg.underlyingDist.pdf(n*0.05 - 2.5)))
spline(w.toIterable)

unhold()
legend(List("UESN", "Gaussian", "Skew Gaussian"))
title("Probability Density Functions")

//push forward maps
implicit val detImpl: DataPipe[Double, Double] = DataPipe(math.abs)

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

println("E[Q] = "+E(q))
