{
  import io.github.mandar2812.dynaml.tensorflow._
  import io.github.mandar2812.dynaml.tensorflow.dynamics._
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.learn.Mode
  import org.platanios.tensorflow.api.learn.layers.Layer
  import org.platanios.tensorflow.api.tensors.TensorLike

  import scala.collection.mutable.ArrayBuffer
  import scala.util.Random


  implicit val mode: Mode = tf.learn.INFERENCE

  val random = new Random()

  val input_dim = 3

  val output_dims = 3

  def batch(dim: Int, batchSize: Int): Tensor = {
    val inputs = ArrayBuffer.empty[Seq[Float]]
    var i = 0
    while (i < batchSize) {
      val input = Seq.tabulate(dim)(_ => random.nextFloat())
      inputs += input
      i += 1
    }

    Tensor(inputs).reshape(Shape(-1, input_dim))
  }

  val inputs = tf.placeholder(FLOAT32, Shape(-1, input_dim))

  val layer = new Layer[Output, Output]("Exp") {
    override val layerType = "Exp"

    override protected def _forward(input: Output)(implicit mode: Mode): Output =
      input.exp
  }

  val feedforward = new Layer[Output, Output]("Exp") {
    override val layerType = "MatMul"

    override protected def _forward(input: Output)(implicit mode: Mode): Output =
      tf.matmul(
        input,
        dtf.tensor_f32(input.shape(1), output_dims)(
          Seq.tabulate(input.shape(1), output_dims)((i, j) => if(i == j) 1.0 else 0.5).flatten:_*
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

  val trainBatch = batch(dim = input_dim, batchSize = 1)
  val feeds = Map(inputs -> trainBatch)

  val results: (Tensor, Tensor, TensorLike, TensorLike) =
    session.run(feeds = feeds, fetches = (outputs, inter_outputs, grad, hess))

  println("Input\n")

  println(trainBatch.summarize(100, false)+"\n")

  println("Outputs\n")

  println(results._1.summarize(100, false)+"\n")

  println()

  println("Intermediate Outputs\n")

  println(results._2.toTensor.summarize(100, false)+"\n")

  println("Jacobian\n")
  println(results._3.toTensor.summarize(100, false)+"\n")

  println("Hessian\n")
  println(results._4.toTensor.summarize(100, false)+"\n")

}
