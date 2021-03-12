import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.analysis._
import io.github.tailhq.dynaml.graphics.charts.Highcharts._
import io.github.tailhq.dynaml.DynaMLPipe._

val data = Array((-1d, 2.5), (0.5, 2.5), (0.75, 3.5), (0.8d, 8.5), (1d, 8.5), (4d, 6.5), (5d, 3.5), (6d, 2.5))
val (knots, values) = data.unzip
val xs = numeric_range(-1d, 6d)(500)

val sorted_knots = knots.sorted


val spline_generator = SplineGenerator(knots, DenseVector(values))
//val spline_generator = SplineGenerator(gen)(knots.indices, DenseVector(values))


scatter(data.toSeq)
hold()
spline(xs.map(x => (x, spline_generator(3)(x))))
unhold()
