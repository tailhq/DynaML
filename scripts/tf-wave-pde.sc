import scala.util.Random
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow._
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

@main
def main(size: Int = 500, num_iterations: Int = 1000) = {

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
  val U_ = U + eps * Ut
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
      feeds = Map(eps -> Tensor(0.03f), damping -> Tensor(0.04f)),
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

