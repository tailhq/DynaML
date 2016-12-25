import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PushforwardMap}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.GaussianRV
import spire.implicits._
import com.quantifind.charts.Highcharts._
/**
  * @author mandar date 22/12/2016.
  */

val g = GaussianRV(0.0, 0.25)

implicit val detImpl = identityPipe[Double]

val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.exp(x)),
  DifferentiableMap(
    (x: Double) => math.log(x),
    (x: Double) => 1.0/x)
)

val p = h->g

val x = Array.tabulate[(Double, Double)](100)(n => (n*0.03, p.underlyingDist.pdf(n*0.03)))
spline(x.toIterable)
