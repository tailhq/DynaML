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
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, RandomUniformInitializer}

import scala.util.Random
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._


val random = new Random()

def batch(dim: Int, min: Double, max: Double, gridSize: Int, func: Seq[Double] => Seq[Double]): (Tensor, Tensor) = {

  val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize) :+ max))

  val targets = points.map(func)

  (
    dtf.tensor_from("FLOAT64", Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten),
    dtf.tensor_from("FLOAT64", Seq.fill(dim)(gridSize + 1).product, 3)(targets)
  )
}

val layer = new Layer[Output, Output]("Sin") {
  override val layerType = "Exp"

  override protected def _forward(input: Output)(implicit mode: Mode): Output =
    input.exp
}

def plot_outputs(x: Tensor, t: Tensor, titleS: String): Unit = {

  val size = x.shape(0)

  val num_outputs = t.shape(1)

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs = (0 until num_outputs).map(i => t(row, i).scalar.asInstanceOf[Double])

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

def plot3d_outputs(x: Tensor, t: Tensor): LinePlot3D = {
  val size = x.shape(0)

  val num_outputs = t.shape(1)

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs = (0 until num_outputs).map(i => t(row, i).scalar.asInstanceOf[Double])

    (inputs, outputs)
  })

  val trajectory = data.map(_._2).map(p => (p.head, p(1), p(2)))

  plot3d.draw(trajectory, 2, true)

}


@main
def apply(f1: Double, f2: Double, f3: Double) = {

  val session = Session()


  val tempdir = home/"tmp"

  val summary_dir = tempdir/s"dtf_cascade_ode_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (0.0, 10.0)

  val input_dim: Int = 1

  val output_dim: Int = 3


  val ground_truth = (tl: Seq[Double]) => {

    val t = tl.head

    Seq(
      f1*math.exp(-t/2),
      -2*f1*math.exp(-t/2) + (f2 + 2*f1)*math.exp(-t/4),
      3*f1*math.exp(-t/2)/2 - 3*(f2 + 2*f1)*math.exp(-t/4) + (f3 - 3*f1/2 + 3*(f2 + 2*f1))*math.exp(-t/6)
    )

  }


  val (test_data, test_targets) = batch(input_dim, domain._1, domain._2, gridSize = 10, ground_truth)

  val input = (FLOAT64, Shape(input_dim))

  val output = (Seq(FLOAT64), Seq(Shape(output_dim)))


  val function  =
    dtflearn.feedforward(
      num_units = output_dim, useBias = false,
      RandomUniformInitializer(),
      RandomUniformInitializer())(1) >>
      layer >>
      dtflearn.feedforward(
        num_units = output_dim, useBias = false,
        RandomUniformInitializer(),
        RandomUniformInitializer())(2)



  val training_data = dtfdata.supervised_dataset(
    Iterable(
      (
        dtf.tensor_f64(1)(0.0),
        dtf.tensor_f64(3)(f1, f2, f3)
      )
    )
  )

  val gain = constant[Output](
    name = "Gain",
    Tensor(
      -1.0/2,  0.0,    0.0,
       1.0/2, -1.0/4,  0.0,
       0.0,    1.0/4, -1.0/6).reshape(Shape(3, 3)).transpose()
  )

  val current_state = q("state", function, isSystemVariable = false)

  val cascade_ode = ∇ - (current_state x gain)

  val analysis.GaussianQuadrature(nodes, weights) = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

  val nodes_tensor: Tensor = dtf.tensor_f64(
    nodes.length, 1)(
    nodes:_*
  )

  val weights_tensor: Tensor = dtf.tensor_f64(
    nodes.length)(
    weights:_*
  )

  val cascade_system = dtflearn.dynamical_system[Double](
    Map("water_levels" -> function),
    Seq(cascade_ode), input, output,
    tf.learn.L2Loss("Loss/L2") >> tf.learn.Mean("L2/Mean"),
    nodes_tensor, weights_tensor, Tensor(1.0))


  implicit val seqArgType: Function.ArgType[Seq[Output]] = seqFuncArgTypeConstructor(3)

  val brine_model = cascade_system.solve(
    Seq(training_data),
    dtflearn.model.trainConfig(
      summary_dir,
      tf.train.RMSProp(0.01),
      dtflearn.abs_loss_change_stop(0.0001, 50000),
      Some(
        dtflearn.model._train_hooks(
          summary_dir, stepRateFreq = 1000,
          summarySaveFreq = 1000,
          checkPointFreq = 1000)
      )),
    dtflearn.model.data_ops(training_data.size, training_data.size, 10)
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



  (cascade_system, brine_model, training_data, mae,
    plot3d_outputs(test_data, predictions),
    plot3d_outputs(test_data, test_targets))

}
