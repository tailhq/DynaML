import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.pipes.TupleIntegerEncoder
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.DelauneySurface
import _root_.io.github.mandar2812.dynaml.tensorflow.dtf
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer

//Transform a 2d sequence into a tensor.
def generate_kernel(k: Seq[Seq[Double]], id: String): Output[Float] = {
  val (rows, cols) = (k.length, k.head.length)
  val tk = dtf.tensor_f32(rows, cols, 1, 1)(k.flatten.map(_.toFloat):_*)
  tf.constant(tk, name = id)
}

//A simplified 2D convolution operation
def simple_conv(x: Output[Float], kernel: Output[Float]): Output[Float] = {
  val expanded_x = tf.expandDims(tf.expandDims(x, 0), -1)
  val intermediate_result = tf.conv2D(expanded_x, kernel, 1, 1, SameConvPadding)
  intermediate_result(0, ::, ::, 0)
}

//Compute the 2D laplacian of an array
def laplace(x: Output[Float]): Output[Float] = {
  val laplace_k = generate_kernel(
    Seq(Seq(0.5, 1.0, 0.5), Seq(1.0, -6.0, 1.0), Seq(0.5, 1.0, 0.5)),
    "laplacian_op")
  simple_conv(x, laplace_k)
}

//The mexican hat wavelet function
val mexican  = (sigma: Double) => (x: Double, y: Double) => {
  (1.0/sigma*sigma*math.Pi)*(1.0 - 0.5*(x*x + y*y)/(sigma*sigma))*math.exp(-0.5*(x*x + y*y)/(sigma*sigma))
}

val gaussian = (sigma: Double) => (x: Double, y: Double) => {
  (1.0/sigma*math.sqrt(2*math.Pi))*math.exp(-0.5*(x*x + y*y)/(sigma*sigma))
}

//Plot a snapshot of the solution as a 3d plot.
def plot_field_snapshot(
  t: Tensor[Float],
  xDomain: (Double, Double),
  yDomain: (Double, Double)): DelauneySurface = {

  val (rows, cols) = (t.shape(0), t.shape(1))

  val stencil_y = utils.range(min = xDomain._1, max = xDomain._2, rows)
  val stencil_x = utils.range(min = yDomain._1, max = yDomain._2, cols)

  val data = t.entriesIterator
    .map(_.asInstanceOf[Float].toDouble)
    .toSeq.grouped(cols).zipWithIndex
    .flatMap(rowI => {
      val (row, row_index) = rowI

      val y = stencil_y(row_index)

      row.zipWithIndex.map(nI => {
        val (num, index) = nI
        val x = stencil_x(index)
        ((x, y), num)
      })
    }).toStream

  plot3d.draw(data)
}

def plot_field(
  solution: Seq[(Tensor[Float], Tensor[Float])])(
  num_snapshots: Int,
  quantity: String = "displacement",
  xDomain: (Double, Double) = (-5.0, 5.0),
  yDomain: (Double, Double) = (-5.0, 5.0)): Seq[DelauneySurface] = {

  val indices = utils.range(1.0, solution.length.toDouble, num_snapshots).map(_.toInt)

  indices.map(i => plot_field_snapshot(
    if(quantity == "displacement") solution(i)._1 else solution(i)._2,
    xDomain, yDomain)
  )
}

@main
def apply(
  size: Int = 75,
  num_iterations: Int = 500,
  eps: Float = 0.001f,
  damping: Float = 0.04f,
  xDomain: (Double, Double) = (-5.0, 5.0),
  yDomain: (Double, Double) = (-5.0, 5.0),
  u_0: (Double, Double) => Double = mexican(1.0)): Seq[(Tensor[Float], Tensor[Float])] = {

  //Start Tensor[Float]flow session
  val sess = Session()

  val (x_grid, y_grid) = (
    utils.range(xDomain._1, xDomain._2, size),
    utils.range(yDomain._1, yDomain._2, size))

  val encoder = TupleIntegerEncoder(List(size, size))

  //Initial Conditions -- some rain drops hit a pond
  val (u_init, ut_init) = (
    Seq.tabulate[Double](size*size)(k => {
      val List(i, j) = encoder.i(k)

      val (x, y) = (x_grid(i), y_grid(j))

      u_0(x, y)
    }),
    Seq.fill[Double](size*size)(0d)
  )

  /*
  * Parameters:
  *
  * eps     -- time resolution
  * damping -- wave damping
  * */
  val eps     = tf.placeholder[Float](Shape(), name = "dt")
  val damping = tf.placeholder[Float](Shape(), name = "damping")

  //Create variables for simulation state
  val U  = tf.variable[Float](
    name = "u", Shape(size, size),
    initializer = ConstantInitializer(dtf.tensor_f32(size, size)(u_init.map(_.toFloat):_*)))

  val Ut = tf.variable[Float](
    name = "ut", Shape(size, size),
    initializer = ConstantInitializer(dtf.tensor_f32(size, size)(ut_init.map(_.toFloat):_*)))

  //Discretized PDE update rules
  val U_  = U + eps * Ut
  val Ut_ = Ut + eps * (laplace(U) - damping * Ut)

  //Operation to update the state
  val step = tf.group(
    Set(U.assign(U_), Ut.assign(Ut_))
  )

  sess.run(targets = tf.globalVariablesInitializer())

  println(s"Solving wave PDE for ${num_iterations} time steps")
  val pb = new utils.ProgressBar(num_iterations)
  //Run num_iterations steps of PDE
  (1 to num_iterations).map(i => {
    //print("Iteration ")
    //pprint.pprintln(i)

    val step_output: (Tensor[Float], Tensor[Float]) = sess.run(
      feeds = Map(eps -> Tensor[Float](0.01f), damping -> Tensor[Float](0.04f)),
      fetches = (U_, Ut_),
      targets = step)

    //print("Maximum Wave Displacement = ")
    //pprint.pprintln(step_output._1.max().scalar.asInstanceOf[Float])
    //println()

    //print("Maximum Wave Velocity = ")
    //pprint.pprintln(step_output._2.max().scalar.asInstanceOf[Float])
    //println()

    pb += 1
    step_output
  })
}

