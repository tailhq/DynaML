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
    Seq.tabulate(dim)(
      i =>
        if (min(i) != max(i)) utils.range(min(i), max(i), gridSize) :+ max(i)
        else Stream(min(i))
    )
  )

  val targets = points.map(func)

  val data_size = points.length

  (
    dtf
      .tensor_from[T](data_size, dim)(points.flatten.toSeq),
    dtf.tensor_from[T](data_size, 1)(targets)
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

  val data = (0 until x.shape(0)).map(row => {

    val inputs = (
      x(row, 0).scalar.asInstanceOf[Double],
      x(row, 1).scalar.asInstanceOf[Double]
    )
    val output = t(row).scalar.asInstanceOf[Double]

    (inputs, output)
  })

  plot3d.draw(data)
}

def plot_field_snapshots(
  x: Tensor[Float],
  t: Tensor[Float]
): Seq[(Float, DelauneySurface)] = {

  x.unstack(axis = 0)
    .zip(t.unstack(axis = 0))
    .map(p => {
      val t_index = p._1(0)
      val xy      = p._1(1 ::).entriesIterator.toSeq
      (
        t_index.scalar.toFloat,
        ((xy.head.toDouble, xy.last.toDouble), p._2.scalar.toDouble)
      )
    })
    .groupBy(_._1)
    .mapValues(_.map(_._2))
    .mapValues(v => plot3d.draw(v))
    .toSeq
    .sortBy(_._1)

}
@main
def apply(
  num_data: Int = 100,
  num_colocation_points: Int = 10000,
  num_neurons: Seq[Int] = Seq(5, 5),
  optimizer: Optimizer = tf.train.Adam(0.01f),
  iterations: Int = 50000,
  reg: Double = 0.001,
  reg_sources: Double = 0.001,
  pde_wt: Double = 1.5,
  vis: Double = 0.5,
  q_scheme: String = "GL",
  tempdir: Path = home / "tmp"
) = {

  val session = Session()

  val summary_dir = tempdir / s"dtf_burgers_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (-5.0, 5.0)

  val time_domain = (0d, 20d)

  val input_dim: Int = 2

  val output_dim: Int = 1

  val nu = vis.toFloat

  val u = 2.5f

  val ground_truth = (tl: Seq[Float]) => {
    val (t, x) = (tl.head, tl.last)

    u * math.exp(-x * x / 0.5f).toFloat
  }

  val f1 = (l: Double) => u * math.exp(-l.toFloat * l.toFloat / 0.5f).toFloat

  val (test_data, _) = batch[Float](
    input_dim,
    Seq(time_domain._1.toFloat, domain._1.toFloat),
    Seq(time_domain._2.toFloat, domain._2.toFloat),
    gridSize = 20,
    ground_truth
  )

  val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2) ++ Seq(
    0d
  )

  val training_data =
    dtfdata.supervised_dataset[Tensor[Float], Tensor[Float]](
      data = xs.flatMap(x => {
        Seq(
          (
            dtf.tensor_f32(input_dim)(0f, x.toFloat),
            dtf.tensor_f32(output_dim)(f1(x))
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
    )(num_neurons ++ Seq(1))

  val (_, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      d = input_dim,
      num_pred_dims = 1,
      num_neurons
    )

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes =
    layer_parameter_names.map(n => "")

  val viscosity =
    constant[Output[Float], Float]("viscosity", Tensor(nu).reshape(Shape()))

  val burgers_equation = d_t + (d_s * I[Float, Float]()) - (d_s(d_s) * viscosity)

  val (quadrature_space, quadrature_time) = if (q_scheme == "MonteCarlo") {
    (
      analysis.monte_carlo_quadrature(
        UniformRV(domain._1, domain._2)
      )(math.sqrt(num_colocation_points).toInt),
      analysis.monte_carlo_quadrature(
        UniformRV(time_domain._1, time_domain._2)
      )(math.sqrt(num_colocation_points).toInt)
    )
  } else {
    (
      analysis.eightPointGaussLegendre.scale(domain._1, domain._2),
      analysis.eightPointGaussLegendre.scale(time_domain._1, time_domain._2)
    )
  }

  val (nodes, weights)     = (quadrature_space.nodes, quadrature_space.weights)
  val (nodes_t, weights_t) = (quadrature_time.nodes, quadrature_time.weights)

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

  val burgers_system1d = dtflearn.pde_system[Float, Float, Float](
    architecture,
    burgers_equation,
    input,
    output,
    loss_func,
    nodes_tensor,
    weights_tensor,
    Tensor(pde_wt.toFloat).reshape(Shape()),
    reg_f = Some(reg_y),
    reg_v = None
  )

  val burgers_model1d = burgers_system1d.solve(
    training_data,
    dtflearn.model.trainConfig(
      summary_dir,
      dtflearn.model.data_ops(
        training_data.size / 10,
        training_data.size / 4,
        50
      ),
      optimizer,
      dtflearn.abs_loss_change_stop(0.001, iterations),
      Some(dtflearn.model._train_hooks(summary_dir))
    ),
    dtflearn.model.tf_data_handle_ops(
      bufferSize = training_data.size / 10,
      patternToTensor = Some(
        burgers_system1d.pattern_to_tensor
      )
    )
  )

  print("Test Data Shapes: ")
  pprint.pprintln(test_data.shape)

  val predictions = burgers_model1d.predict("Output")(test_data)

  val plot = plot_field(test_data, predictions.head)

  session.close()

  (
    burgers_system1d,
    burgers_model1d,
    training_data,
    plot,
    test_data,
    predictions.head
  )

}

def two_d(
  num_data: Int = 100,
  num_colocation_points: Int = 10000,
  num_neurons: Seq[Int] = Seq(5, 5),
  optimizer: Optimizer = tf.train.Adam(0.01f),
  iterations: Int = 50000,
  reg: Double = 0.001,
  reg_sources: Double = 0.001,
  pde_wt: Double = 1.5,
  vis: Double = 0.5,
  q_scheme: String = "GL",
  tempdir: Path = home / "tmp"
) = {

  val session = Session()

  val summary_dir = tempdir / s"dtf_burgers2d_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (-5.0, 5.0)

  val time_domain = (0d, 20d)

  val input_dim: Int = 3

  val output_dim: Int = 1

  val nu = vis.toFloat

  val u = 2.5f

  val ground_truth = (tl: Seq[Float]) => {
    val (t, x) = (tl.head, tl.tail)

    u * math.exp(-x.map(x => x * x).sum / 2f).toFloat
  }

  val f1 = (l: Seq[Double]) => u * math.exp(-l.map(x => x * x).sum / 2f).toFloat //if (l == 0d) u else 0f

  val initial_cond_plot = plot3d.draw(
    (x, y) => f1(Seq(x, y)).toFloat,
    (domain._1.toFloat, domain._2.toFloat),
    (domain._1.toFloat, domain._2.toFloat)
  )

  val (test_data, _) = batch[Float](
    input_dim,
    Seq(time_domain._1.toFloat, domain._1.toFloat, domain._1.toFloat),
    Seq(time_domain._2.toFloat, domain._2.toFloat, domain._2.toFloat),
    gridSize = 10,
    ground_truth
  )

  val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2) ++ Seq(
    0d
  )

  val xsys = utils.combine(Seq(xs, xs))

  val training_data =
    dtfdata.supervised_dataset[Tensor[Float], Tensor[Float]](
      data = xsys.flatMap(xy => {
        Seq(
          (
            dtf.buffer_f32(
              Shape(input_dim),
              Array(0f) ++ xy.map(_.toFloat).toArray[Float]
            ),
            dtf.tensor_f32(output_dim)(f1(xy))
          )
        )
      })
    )

  val input = Shape(input_dim)

  val output = Shape(output_dim)

  val architecture =
    dtflearn.rbf_layer[Float](
      "RBF",
      num_neurons.head,
      dtflearn.rbf_layer.InverseMultiQuadric
    ) >>
      dtflearn.feedforward_stack[Float](
        (i: Int) =>
          if (i == 1) tf.learn.ReLU(s"Act_$i", 0.01f)
          else tf.learn.Sigmoid(s"Act_$i")
      )(num_neurons.tail ++ Seq(1))

  val (_, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      d = num_neurons.head,
      num_pred_dims = 1,
      num_neurons.tail
    )

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes =
    layer_parameter_names.map(n => "")

  val viscosity =
    constant[Output[Float], Float]("viscosity", Tensor(nu).reshape(Shape()))

  val d_x = ∂[Float]("D_x")(1)(---)
  val d_y = ∂[Float]("D_y")(2)(---)

  val burgers_equation = d_t + (d_x * I[Float, Float]()) - (d_x(d_x) + d_y(d_y)) * viscosity

  val (quadrature_space, quadrature_time) = if (q_scheme == "MonteCarlo") {
    val time_n = UniformRV(time_domain._1, time_domain._2)
      .iid(math.pow(num_colocation_points, 1d / input_dim).toInt)
      .draw
      .toSeq
    val space_n = UniformRV(domain._1, domain._2)
      .iid(math.pow(num_colocation_points, 1d / input_dim).toInt)
      .draw
      .toSeq

    (
      analysis.MonteCarloQuadrature(space_n),
      analysis.MonteCarloQuadrature(time_n)
    )
  } else {
    (
      analysis.eightPointGaussLegendre.scale(domain._1, domain._2),
      analysis.eightPointGaussLegendre.scale(time_domain._1, time_domain._2)
    )
  }

  val (nodes, weights)     = (quadrature_space.nodes, quadrature_space.weights)
  val (nodes_t, weights_t) = (quadrature_time.nodes, quadrature_time.weights)

  val nodes_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length * nodes.length * nodes_t.length,
    input_dim
  )(
    utils
      .combine(
        Seq(nodes_t.map(_.toFloat), nodes.map(_.toFloat), nodes.map(_.toFloat))
      )
      .flatten: _*
  )

  val weights_tensor: Tensor[Float] =
    dtf.tensor_f32(nodes.length * nodes.length * nodes_t.length)(
      utils
        .combine(
          Seq(
            weights_t.map(_.toFloat),
            weights.map(_.toFloat),
            weights.map(_.toFloat)
          )
        )
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

  val burgers_system2d = dtflearn.pde_system[Float, Float, Float](
    architecture,
    burgers_equation,
    input,
    output,
    loss_func,
    nodes_tensor,
    weights_tensor,
    Tensor(pde_wt.toFloat).reshape(Shape()),
    reg_f = Some(reg_y),
    reg_v = None
  )

  val burgers_model2d = burgers_system2d.solve(
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
      patternToTensor = Some(
        burgers_system2d.pattern_to_tensor
      )
    )
  )

  print("Test Data Shapes: ")
  pprint.pprintln(test_data.shape)

  val predictions = burgers_model2d.predict("Output")(test_data)

  val r = predictions.head.rank
  val plot = plot_field_snapshots(
    test_data,
    if (r > 2) predictions.head(0) else predictions.head
  )

  session.close()

  (
    initial_cond_plot,
    burgers_system2d,
    burgers_model2d,
    training_data,
    plot,
    test_data,
    if (r > 2) predictions.head(0) else predictions.head
  )

}
