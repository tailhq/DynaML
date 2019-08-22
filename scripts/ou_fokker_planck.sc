import _root_.io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}
import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.graphics.plot3d._
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{
  L1Regularization,
  L2Regularization
}
import _root_.io.github.mandar2812.dynaml.tensorflow.pde.{source => q, _}
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import ammonite.ops.home
import org.joda.time.DateTime
import _root_.org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import spire.algebra.InnerProductSpace

import scala.util.Random

val random = new Random()

def batch[T: TF: IsFloatOrDouble](
  dim: Int,
  min: Seq[T],
  max: Seq[T],
  gridSize: Int,
  func: Seq[T] => T
)(
  implicit f: InnerProductSpace[T, Double]
): (Tensor[T], Tensor[T]) = {

  val points = utils.combine(
    Seq.tabulate(dim)(i => utils.range(min(i), max(i), gridSize) :+ max(i))
  )

  val targets = points.map(func)

  (
    dtf
      .tensor_from[T](Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten),
    dtf.tensor_from[T](Seq.fill(dim)(gridSize + 1).product, 1)(targets)
  )
}

val layer = new Layer[Output[Float], Output[Float]]("Sin") {
  override val layerType = "Sin"

  override def forwardWithoutContext(
    input: Output[Float]
  )(
    implicit mode: Mode
  ): Output[Float] =
    tf.sin(input)
}

def plot_field(x: Tensor[Float], t: Tensor[Float]): DelauneySurface = {

  val size = x.shape(0)

  val data = (0 until size).map(row => {

    val inputs = (
      x(row, 0).scalar.asInstanceOf[Double],
      x(row, 1).scalar.asInstanceOf[Double]
    )
    val output = t(row).scalar.asInstanceOf[Double]

    (inputs, output)
  })

  plot3d.draw(data)
}

@main
def apply(
  num_data: Int = 100,
  num_neurons: Seq[Int] = Seq(5, 5),
  optimizer: Optimizer = tf.train.Adam(0.01f),
  iterations: Int = 50000,
  reg: Double = 0.001,
  reg_sources: Double = 0.001,
  pde_wt: Double = 1.5
) = {

  val session = Session()

  val tempdir = home / "tmp"

  val summary_dir = tempdir / s"dtf_fokker_planck_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (-5.0, 5.0)

  val time_domain = (0d, 10d)

  val input_dim: Int = 2

  val output_dim: Int = 1

  val diff = 1.5
  val x0   = domain._2 - 1.5d
  val th   = 0.25

  val ground_truth = (tl: Seq[Float]) => {
    val (t, x) = (tl.head, tl.last)

    (math.sqrt(th / (2 * math.Pi * diff * (1 - math.exp(-2 * th * t)))) *
      math.exp(
        -1 * th * math
          .pow(x - x0 * math.exp(-th * t), 2) / (2 * diff * (1 - math
          .exp(-2 * th * t)))
      )).toFloat
  }

  val f1 = (l: Double) => if (l == x0) 1d else 0d

  val (test_data, test_targets) = batch[Float](
    input_dim,
    Seq(time_domain._1.toFloat, domain._1.toFloat),
    Seq(time_domain._2.toFloat, domain._2.toFloat),
    gridSize = 20,
    ground_truth
  )

  val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2)

  val rv = UniformRV(time_domain._1, time_domain._2)

  val training_data =
    dtfdata.supervised_dataset[Tensor[Float], Tensor[Float]](
      data = xs.flatMap(x => {

        val rand_time = rv.draw.toFloat

        Seq(
          (
            dtf.tensor_f32(input_dim)(0f, x.toFloat),
            dtf.tensor_f32(output_dim)(f1(x).toFloat)
          ),
          (
            dtf.tensor_f32(input_dim)(rand_time, x.toFloat),
            dtf.tensor_f32(output_dim)(
              ground_truth(Seq(rand_time.toFloat, x.toFloat))
            )
          )
        )
      })
    )

  val input = Shape(input_dim)

  val output = Shape(output_dim)

  val architecture =
    dtflearn.feedforward_stack[Float](
      (i: Int) =>
        if (i == 1) tf.learn.ReLU(s"Act_$i") else tf.learn.Sigmoid(s"Act_$i")
    )(num_neurons ++ Seq(1)) >>
      tf.learn.Sigmoid("Act_Output")

  val x_layer = dtflearn.layer(
    name = "X",
    MetaPipe[Mode, Output[Float], Output[Float]](_ => xt => xt(---, 1 ::))
  )

  val theta_layer = tf.learn.Linear[Float]("theta", 1) >> dtflearn.layer(
    name = "Act_theta",
    MetaPipe[Mode, Output[Float], Output[Float]](_ => xt => xt.pow(2))
  )
  val diff_layer = tf.learn.Linear[Float]("D", 1) >> dtflearn.layer(
    name = "Act_D",
    MetaPipe[Mode, Output[Float], Output[Float]](_ => xt => xt.pow(2))
  )

  val (_, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      d = input_dim,
      num_pred_dims = 1,
      num_neurons
    )

  val (th_parameters, th_shapes, th_dt) =
    (Seq("theta/Weights"), Seq(Shape(input_dim, 1)), Seq("FLOAT32"))
  val (d_parameters, d_shapes, d_dt) =
    (Seq("D/Weights"), Seq(Shape(input_dim, 1)), Seq("FLOAT32"))

  val th_scopes =
    th_parameters.map(_ => "" /*dtfutils.get_scope(theta_layer)*/ )
  val d_scopes = d_parameters.map(_ => "" /*dtfutils.get_scope(diff_layer)*/ )

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes =
    layer_parameter_names.map(n => "" /*scope(n.split("/").last)*/ )

  val theta = q[Output[Float], Float](
    name = "theta",
    theta_layer,
    isSystemVariable = true
  )

  val D =
    q[Output[Float], Float](name = "D", diff_layer, isSystemVariable = true)

  val x = q[Output[Float], Float](name = "X", x_layer, isSystemVariable = false)

  val ornstein_ulhenbeck: Operator[Output[Float], Output[Float]] =
    d_t - theta * d_s(x * I[Float, Float]()) - d_s(D * d_s)

  val analysis.GaussianQuadrature(nodes, weights) =
    analysis.eightPointGaussLegendre.scale(domain._1, domain._2)
  val analysis.GaussianQuadrature(nodes_t, weights_t) =
    analysis.eightPointGaussLegendre.scale(time_domain._1, time_domain._2)

  val nodes_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length * nodes_t.length,
    2
  )(
    utils.combine(Seq(nodes_t.map(_.toFloat), nodes.map(_.toFloat))).flatten: _*
  )

  val weights_tensor: Tensor[Float] =
    dtf.tensor_f32(nodes.length * nodes_t.length)(
      utils
        .combine(Seq(weights_t.map(_.toFloat), weights.map(_.toFloat)))
        .map(_.product): _*
    )

  val loss_func =
    tf.learn.L2Loss[Float, Float]("Loss/L2") >>
      tf.learn.Mean[Float]("L2/Mean")

  val reg_y = L2Regularization[Float](
    layer_scopes,
    layer_parameter_names,
    layer_datatypes,
    layer_shapes,
    reg
  )

  val reg_v = L1Regularization[Float](
    th_scopes ++ d_scopes,
    th_parameters ++ d_parameters,
    th_dt ++ d_dt,
    th_shapes ++ d_shapes,
    reg_sources
  )

  val wave_system1d = dtflearn.pde_system[Float, Float, Float](
    architecture,
    ornstein_ulhenbeck,
    input,
    output,
    loss_func,
    nodes_tensor,
    weights_tensor,
    Tensor(pde_wt.toFloat).reshape(Shape()),
    reg_f = Some(reg_y),
    reg_v = Some(reg_v)
  )

  val wave_model1d = wave_system1d.solve(
    training_data,
    dtflearn.model.trainConfig(
      summary_dir,
      dtflearn.model.data_ops(
        training_data.size / 10,
        training_data.size / 4,
        10
      ),
      optimizer,
      dtflearn.abs_loss_change_stop(0.001, iterations),
      Some(dtflearn.model._train_hooks(summary_dir))
    ),
    dtflearn.model.tf_data_handle_ops(
        bufferSize = training_data.size / 10,
        patternToTensor = Some(wave_system1d.pattern_to_tensor)
    )
  )

  print("Test Data Shapes: ")
  pprint.pprintln(test_data.shape)
  pprint.pprintln(test_targets.shape)

  val predictions = wave_model1d.predict("Output", "theta", "D")(test_data)

  val plot = plot_field(test_data, predictions.head)

  val plot_th = plot_field(test_data, predictions(1))

  val plot_d = plot_field(test_data, predictions(2))

  val error_tensor = tfi.subtract(predictions.head, test_targets)

  val mae = tfi.mean(tfi.abs(error_tensor)).scalar

  session.close()

  print("Test Error is = ")
  pprint.pprintln(mae)

  val error_plot = plot_field(test_data, error_tensor)

  (
    wave_system1d,
    wave_model1d,
    training_data,
    plot,
    error_plot,
    plot_th,
    plot_d,
    mae
  )

}
