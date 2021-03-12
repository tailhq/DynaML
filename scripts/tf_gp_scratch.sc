import _root_.io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.org.platanios.tensorflow.api.ops.variables.{
  Initializer,
  RandomNormalInitializer
}

class SEKernelTF[O: TF: IsFloatOrDouble](
  override val name: String,
  sigmaInitializer: Initializer = tf.RandomNormalInitializer(),
  lengthScaleInitializer: Initializer = tf.RandomNormalInitializer())
    extends Layer[(Output[O], Output[O]), Output[O]](name) {

  override val layerType: String = s"RBFKernel"

  override def forwardWithoutContext(
    input: (Output[O], Output[O])
  )(
    implicit mode: Mode
  ): Output[O] = {

    val variance = tf.variable[O]("variance", Shape(), sigmaInitializer)

    val lengthScale =
      tf.variable[O]("lengthscale", Shape(), lengthScaleInitializer)

    val (xs, ys) = (input._1, input._2)

    tf.stack(
      tf.unstack(ys, axis = 0)
        .map(y => {
          val d = xs.subtract(y)
          SEKernelTF.compute(d, variance, lengthScale)
        }),
      axis = 1
    )

  }
}

object SEKernelTF {

  def compute[O: TF: IsFloatOrDouble](
    d: Output[O],
    s: Output[O],
    l: Output[O]
  ): Output[O] =
    tf.multiply(
      tf.square(s),
      tf.exp(
        tf.divide(
          tf.square(d).sum(axes = 1),
          tf.square(l) * Output(-2).reshape(Shape()).castTo[O]
        )
      )
    )
}

type T = Float
val mean: Layer[Output[T], Output[T]] = tf.learn.Linear[Float]("MeanFunc", 1)
val kernel: Layer[(Output[T], Output[T]), Output[T]] =
  new SEKernelTF[T]("Kernel")

val sess          = Session()
val uniform_float = UniformRV(0d, 2d) > DataPipe((x: Double) => x.toFloat)

val num_dim     = 3
val batch1_size = 5
val batch2_size = 3

val batch1 = dtf.random[Float](batch1_size, num_dim)(
  uniform_float
)

val batch2 = dtf.random[Float](batch2_size, num_dim)(
  uniform_float
)

batch1.summarize()
batch2.summarize()

implicit val mode: Mode = tf.learn.INFERENCE

val inputs = (
  tf.placeholder[Float](Shape(batch1_size, num_dim)),
  tf.placeholder[Float](Shape(batch2_size, num_dim))
)

sess.run(targets = tf.globalVariablesInitializer())

val crosskernelMat =
  sess.run(
    feeds = Map(inputs._1 -> batch1, inputs._2 -> batch2),
    fetches = kernel(inputs)
  )

val kernelMat = sess.run(
  feeds = Map(inputs._1 -> batch1),
  fetches = kernel((inputs._1, inputs._1))
)
