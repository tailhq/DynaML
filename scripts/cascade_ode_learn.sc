import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.LinePlot3D
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde.{source => q, _}
import _root_.org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import ammonite.ops.home
import org.joda.time.DateTime
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer
import _root_.io.github.mandar2812.dynaml.tensorflow.implicits._
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.variables.{
  Initializer,
  RandomNormalInitializer,
  RandomUniformInitializer
}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import scala.util.Random
import _root_.io.github.mandar2812.dynaml.graphics.charts.Highcharts._

val random = new Random()

def batch(
  dim: Int,
  min: Double,
  max: Double,
  gridSize: Int,
  func: Seq[Double] => Seq[Double]
): (Tensor[Float], Tensor[Float]) = {

  val points =
    utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize) :+ max))

  val targets = points.map(func)

  (
    dtf.tensor_from[Float](Seq.fill(dim)(gridSize + 1).product, dim)(
      points.flatten.map(_.toFloat)
    ),
    dtf.tensor_from[Float](Seq.fill(dim)(gridSize + 1).product, 3)(
      targets.flatten.map(_.toFloat)
    )
  )
}

val layer = new Layer[Output[Float], Output[Float]]("Sin") {
  override val layerType = "Exp"

  override def forwardWithoutContext(
    input: Output[Float]
  )(
    implicit mode: Mode
  ): Output[Float] =
    input.exp
}

def plot_outputs(x: Tensor[Float], t: Tensor[Float], titleS: String): Unit = {

  val size = x.shape(0)

  val num_outputs = t.shape(1)

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs =
      (0 until num_outputs).map(i => t(row, i).scalar.asInstanceOf[Double])

    (inputs, outputs)
  })

  spline(data.map(p => (p._1, p._2.head)))
  hold()
  spline(data.map(p => (p._1, p._2(1))))
  spline(data.map(p => (p._1, p._2.last)))
  legend(Seq("Output 1", "Output 2", "Output 3"))
  xAxis("Time")
  yAxis("Output")
  unhold()
  title(titleS)
}

def plot3d_outputs(x: Tensor[Float], t: Tensor[Float]): LinePlot3D = {
  val size = x.shape(0)

  val num_outputs = t.shape(1)

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs =
      (0 until num_outputs).map(i => t(row, i).scalar.asInstanceOf[Double])

    (inputs, outputs)
  })

  val trajectory = data.map(_._2).map(p => (p.head, p(1), p(2)))

  plot3d.draw(trajectory, 2, true)

}
@main
def apply(
  f1: Double,
  f2: Double,
  f3: Double,
  num_neurons: Seq[Int] = Seq(5, 5),
  optimizer: Optimizer = tf.train.Adam(0.01f),
  iterations: Int = 50000,
  q_scheme: String = "GL",
  tempdir: Path = home / "tmp"
) = {

  val session = Session()

  val tempdir = home / "tmp"

  val summary_dir = tempdir / s"dtf_cascade_ode_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (0.0, 10.0)

  val input_dim: Int = 1

  val output_dim: Int = 3

  val ground_truth = (tl: Seq[Double]) => {

    val t = tl.head

    Seq(
      f1 * math.exp(-t / 2),
      -2 * f1 * math.exp(-t / 2) + (f2 + 2 * f1) * math.exp(-t / 4),
      3 * f1 * math.exp(-t / 2) / 2 - 3 * (f2 + 2 * f1) * math
        .exp(-t / 4) + (f3 - 3 * f1 / 2 + 3 * (f2 + 2 * f1)) * math.exp(-t / 6)
    )

  }

  val (test_data, test_targets) =
    batch(input_dim, domain._1, domain._2, gridSize = 10, ground_truth)

  val input = Shape(input_dim)

  val output = Shape(output_dim)

  val function =
    dtflearn.feedforward_stack[Float](
      (i: Int) =>
        if (i == 1) tf.learn.ReLU(s"Act_$i") else tf.learn.Sigmoid(s"Act_$i")
    )(num_neurons ++ Seq(output_dim)) >>
      tf.learn.Sigmoid("Act_Output")

  val training_data = dtfdata.supervised_dataset(
    Iterable(
      (
        dtf.tensor_f32(input_dim)(0.0f),
        dtf.tensor_f32(output_dim)(f1.toFloat, f2.toFloat, f3.toFloat)
      )
    )
  )

  val gain = constant[Output[Float], Float](
    name = "Gain",
    Tensor[Float](
      -1.0f / 2,
      0.0f,
      0.0f,
      1.0f / 2,
      -1.0f / 4,
      0.0f,
      0.0f,
      1.0f / 4,
      -1.0f / 6
    ).reshape(Shape(3, 3))
      .transpose()
  )

  val cascade_ode = ∇ - (I[Float, Float]() x gain)

  val analysis.GaussianQuadrature(nodes, weights) =
    analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

  val nodes_tensor: Tensor[Float] = dtf.tensor_f32(nodes.length, 1)(
    nodes.map(_.toFloat): _*
  )

  val weights_tensor: Tensor[Float] = dtf.tensor_f32(nodes.length)(
    weights.map(_.toFloat): _*
  )

  val cascade_system = dtflearn.pde_system[Float, Float, Float](
    function,
    cascade_ode,
    input,
    output,
    tf.learn.L2Loss[Float, Float]("Loss/L2") >> tf.learn.Mean[Float]("L2/Mean"),
    nodes_tensor,
    weights_tensor,
    Tensor[Float](1.0f).reshape(Shape()),
    name = "water_levels"
  )

  val brine_model = cascade_system.solve(
    training_data,
    dtflearn.model.trainConfig(
      summary_dir,
      dtflearn.model.data_ops(
        training_data.size / 10,
        training_data.size,
        50
      ),
      optimizer,
      dtflearn.abs_loss_change_stop(0.0001, iterations),
      Some(
        dtflearn.model._train_hooks(summary_dir)
      )
    ),
    dtflearn.model.tf_data_handle_ops(
      bufferSize = 1,
      patternToTensor = Some(
        cascade_system.pattern_to_tensor
      )
    )
  )

  print("Test Data Shapes: ")
  pprint.pprintln(test_data.shape)
  pprint.pprintln(test_targets.shape)

  val predictions = brine_model.predict("water_levels")(test_data).head

  val error_tensor = predictions.subtract(test_targets)

  val mae = error_tensor.abs.mean().scalar.asInstanceOf[Double]

  session.close()

  print("Test Error is = ")
  pprint.pprintln(mae)

  plot_outputs(test_data, predictions, "Predicted Targets")

  plot_outputs(test_data, test_targets, "Actual Targets")

  (
    cascade_system,
    brine_model,
    training_data,
    mae,
    plot3d_outputs(test_data, predictions),
    plot3d_outputs(test_data, test_targets)
  )

}
