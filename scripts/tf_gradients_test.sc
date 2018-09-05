import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde._
import _root_.org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.org.platanios.tensorflow.api.tensors.TensorLike
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


val random = new Random()

def batch(dim: Int, batchSize: Int): Tensor = {
  val inputs = ArrayBuffer.empty[Seq[Float]]
  var i = 0
  while (i < batchSize) {
    val input = Seq.tabulate(dim)(_ => random.nextFloat())
    inputs += input
    i += 1
  }

  Tensor(inputs).reshape(Shape(-1, dim))
}

val layer = new Layer[Output, Output]("Exp") {
  override val layerType = "Exp"

  override protected def _forward(input: Output)(implicit mode: Mode): Output =
    input.exp
}


@main
def apply(input_dim: Int = 3, output_dim: Int = 1, minibatch: Int = 4) = {

  implicit val mode: Mode = tf.learn.INFERENCE

  val inputs = tf.placeholder(FLOAT32, Shape(-1, input_dim))

  val feedforward = new Layer[Output, Output]("Exp") {
    override val layerType = "MatMul"

    override protected def _forward(input: Output)(implicit mode: Mode): Output =
      tf.matmul(
        input,
        dtf.tensor_f32(input.shape(1), output_dim)(
          Seq.tabulate(input.shape(1), output_dim)((i, j) => if(i == j) 1.0 else 10.0).flatten:_*
        )
      )
  }

  val linear_exp = layer >> feedforward

  val inter_outputs = layer.forward(inputs)

  val outputs = linear_exp.forward(inputs)

  val grad    = ∇(linear_exp)(inputs)

  val hess    = hessian(linear_exp)(inputs)

  val session = Session()
  session.run(targets = tf.globalVariablesInitializer())

  val trainBatch = batch(dim = input_dim, batchSize = minibatch)

  val feeds = Map(inputs -> trainBatch)

  val results: (Tensor, Tensor, TensorLike, TensorLike) =
    session.run(feeds = feeds, fetches = (outputs, inter_outputs, grad, hess))

  println("Input\n")

  pprint.pprintln(trainBatch.summarize(100, false)+"\n")

  println("Outputs\n")

  pprint.pprintln(results._1.summarize(100, false)+"\n")

  println()

  println("Intermediate Outputs\n")

  pprint.pprintln(results._2.toTensor.summarize(100, false)+"\n")

  println("Jacobian\n")
  pprint.pprintln(results._3.toTensor.summarize(100, false)+"\n")

  println("Hessian\n")
  pprint.pprintln(results._4.toTensor.summarize(100, false)+"\n")

  results
}
