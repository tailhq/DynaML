import io.github.mandar2812.dynaml.analysis._
val data = Array((-6d, 1.0), (-1d, 0.5), (-0.2, 2.5), (1.2, 0.1), (3d, 1.2), (5d, 5d))
val (knots, values) = data.unzip
val xs = numeric_range(-5d, 5d)(500)

val sorted_knots = knots.sorted


val spline_generator = SplineGenerator(knots, DenseVector(values))
//val spline_generator = SplineGenerator(gen)(knots.indices, DenseVector(values))


scatter(data.toSeq)
hold()
spline(xs.map(x => (x, spline_generator(3)(x))))
