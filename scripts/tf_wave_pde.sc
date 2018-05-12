import scala.util.Random
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.DelauneySurface
import _root_.io.github.mandar2812.dynaml.tensorflow.dtf
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer

//Transform a 2d sequence into a tensor.
def generate_kernel(k: Seq[Seq[Double]], id: String): Output = {
  val (rows, cols) = (k.length, k.head.length)
  val tk = dtf.tensor_f32(rows, cols, 1, 1)(k.flatten:_*)
  tf.constant(tk, name = id)
}

//A simplified 2D convolution operation
def simple_conv(x: Output, kernel: Output): Output = {
  val expanded_x = tf.expandDims(tf.expandDims(x, 0), -1)
  tf.conv2D(expanded_x, kernel, 1, 1, SamePadding)(0, ::, ::, 0)
}

//Compute the 2D laplacian of an array
def laplace(x: Output): Output = {
  val laplace_k = generate_kernel(
    Seq(Seq(0.5, 1.0, 0.5), Seq(1.0, -6.0, 1.0), Seq(0.5, 1.0, 0.5)),
    "laplacian_op")
  simple_conv(x, laplace_k)
}

//Plot a snapshot of the solution as a 3d plot.
def plot_field_snapshot(
  t: Tensor,
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
  solution: Seq[(Tensor, Tensor)])(
  num_snapshots: Int,
  quantity: String = "displacement",
  xDomain: (Double, Double) = (-5.0, 5.0),
  yDomain: (Double, Double) = (-5.0, 5.0)): Seq[DelauneySurface] = {

  val indices = utils.range(1.0, solution.length.toDouble, num_snapshots).map(_.toInt).filter(_ < 1000)

  indices.map(i => plot_field_snapshot(
    if(quantity == "displacement") solution(i)._1 else solution(i)._2,
    xDomain, yDomain)
  )
}

@main
def main(
  size: Int = 500,
  num_iterations: Int = 1000,
  eps: Float = 0.001f,
  damping: Float = 0.04f): Seq[(Tensor, Tensor)] = {

  //Start Tensorflow session
  val sess = Session()

  //Initial Conditions -- some rain drops hit a pond
  val (u_init, ut_init) = (
    Seq.tabulate[Double](size*size)(_ => if(Random.nextDouble() <= 0.95) 0d else Random.nextDouble()),
    Seq.fill[Double](size*size)(0d)
  )

  /*
  * Parameters:
  *
  * eps     -- time resolution
  * damping -- wave damping
  * */
  val eps     = tf.placeholder(FLOAT32, Shape(), name = "dt")
  val damping = tf.placeholder(FLOAT32, Shape(), name = "damping")

  //Create variables for simulation state
  val U  = tf.variable(
    name = "u", FLOAT32,
    initializer = ConstantInitializer(dtf.tensor_f32(size, size)(u_init:_*)))

  val Ut = tf.variable(
    name = "ut", FLOAT32,
    initializer = ConstantInitializer(dtf.tensor_f32(size, size)(ut_init:_*)))

  //Discretized PDE update rules
  val U_  = U + eps * Ut
  val Ut_ = Ut + eps * (laplace(U) - damping * Ut)

  //Operation to update the state
  val step = tf.group(
    Set(U.assign(U_), Ut.assign(Ut_))
  )

  sess.run(targets = tf.globalVariablesInitializer())

  //Run 1000 steps of PDE
  (1 to num_iterations).map(i => {
    print("Iteration ")
    pprint.pprintln(i)

    val step_output: (Tensor, Tensor) = sess.run(
      feeds = Map(eps -> Tensor(0.001f), damping -> Tensor(0.04f)),
      fetches = (U_, Ut_),
      targets = step)

    print("Maximum Wave Displacement = ")
    pprint.pprintln(step_output._1.max().scalar.asInstanceOf[Float])
    println()

    print("Maximum Wave Velocity = ")
    pprint.pprintln(step_output._2.max().scalar.asInstanceOf[Float])
    println()

    step_output
  })
}

