
import scala.util.Random
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer

//Transform a 2d sequence into a tensor.
def generate_kernel(k: Seq[Seq[Double]], id: String) = {
  val (rows, cols) = (k.length, k.head.length)
  val tk = dtf.tensor_f32(rows, cols, 1, 1)(k.flatten:_*)
  tf.constant(tk, name = id)
}

def simple_conv(x: Output, kernel: Output) = {
  val expanded_x = tf.expandDims(tf.expandDims(x, 0), -1)
  tf.conv2D(expanded_x, kernel, 1, 1, SamePadding)(0, ::, ::, 0)
}

def laplace(x: Output) = {
  val laplace_k = generate_kernel(
    Seq(Seq(0.5, 1.0, 0.5), Seq(1.0, -6.0, 1.0), Seq(0.5, 1.0, 0.5)),
    "laplacian_op")
  simple_conv(x, laplace_k)
}

val sess = Session()

val size = 500

val (u_init, ut_init) = (
  Seq.tabulate[Double](size*size)(_ => if(Random.nextDouble() <= 0.95) 0d else Random.nextDouble()),
  Seq.fill[Double](size*size)(0d)
)

val eps     = tf.placeholder(FLOAT32, Shape(), name = "dt")
val damping = tf.placeholder(FLOAT32, Shape(), name = "damping")

val U  = tf.variable(
  name = "u", FLOAT32,
  initializer = ConstantInitializer(dtf.tensor_f32(size, size)(u_init:_*)))

val Ut = tf.variable(
  name = "ut", FLOAT32,
  initializer = ConstantInitializer(dtf.tensor_f32(size, size)(ut_init:_*)))

val U_ = U + eps * Ut
val Ut_ = Ut + eps * (laplace(U) - damping * Ut)

val step = tf.group(
  Set(U.assign(U_), Ut.assign(Ut_))
)

sess.run(targets = tf.globalVariablesInitializer())

val solution = (1 to 1000).map(i => {
  print("Iteration ")
  pprint.pprintln(i)
  println()

  sess.run(
    feeds = Map(eps -> Tensor(0.3f), damping -> Tensor(0.2f)),
    fetches = (U_, Ut_),
    targets = step)
})

