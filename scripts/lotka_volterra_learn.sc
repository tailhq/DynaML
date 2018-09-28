import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.LinePlot3D
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.layers._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde.{source => q, _}
import _root_.org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import ammonite.ops.home
import org.joda.time.DateTime
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer
import _root_.io.github.mandar2812.dynaml.tensorflow.implicits._
import breeze.stats.distributions.Uniform
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, RandomUniformInitializer}

import scala.util.Random
import com.quantifind.charts.Highcharts._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer


val random = new Random()

def batch(dim: Int, min: Double, max: Double, gridSize: Int, func: Seq[Double] => Seq[Double]): (Tensor, Tensor) = {

  val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize) :+ max))

  val targets = points.map(func)

  (
    dtf.tensor_from("FLOAT64", Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten),
    dtf.tensor_from("FLOAT64", Seq.fill(dim)(gridSize + 1).product, 3)(targets)
  )
}

val layer_exp = (n: String) => new Layer[Output, Output](n) {
  override val layerType = "Exp"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    input.exp

  }
}



val layer_trig = (n: String) => new Layer[Output, Output](n) {
  override val layerType = "Cos"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    input.cos
  }
}

def plot_outputs(x: Tensor, t: Seq[Tensor], titleS: String): Unit = {

  val size = x.shape(0)

  val num_outputs = t.length

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs = (0 until num_outputs).map(i => t(i)(row, 0).scalar.asInstanceOf[Double])

    (inputs, outputs)
  })

  spline(data.map(p => (p._1, p._2.head)))
  hold()
  spline(data.map(p => (p._1, p._2(1))))
  legend(Seq("Prey Population", "Predator Population"))
  xAxis("Time")
  yAxis("Output")
  unhold()
  title(titleS)

  spline(data.map(p => (p._2.head, p._2(1))))
  xAxis("Prey Population")
  yAxis("Predator Population")
  title(titleS+": Phase Space Trajectory")
}

def plot3d_outputs(x: Tensor, t: Seq[Tensor]): LinePlot3D = {
  val size = x.shape(0)

  val num_outputs = t.length

  val data = (0 until size).map(row => {

    val inputs = x(row).scalar.asInstanceOf[Double]
    val outputs = (0 until num_outputs).map(i => t(i)(row, 0).scalar.asInstanceOf[Double])

    (inputs, outputs)
  })

  val trajectory = data.map(_._2).map(p => (p.head, p(1), 0.0))

  plot3d.draw(trajectory, 2, true)

}


@main
def apply(
  f1: Double, f2: Double, sizes: Seq[Int] = Seq(20, 20),
  optimizer: Optimizer = tf.train.RMSProp(0.001),
  maxIt: Int = 200000, reg_param: Double = 0.001) = {

  val session = Session()


  val tempdir = home/"tmp"

  val summary_dir = tempdir/s"dtf_lotka_volterra_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (0.0, 200.0)

  val input_dim: Int = 1

  val output_dim: Int = 2


  val ground_truth = (tl: Seq[Double]) => {

    val t = tl.head

    Seq(
      f1*math.exp(-t/2),
      -2*f1*math.exp(-t/2) + (f2 + 2*f1)*math.exp(-t/4),
      3*f1*math.exp(-t/2)/2 - 3*(f2 + 2*f1)*math.exp(-t/4) + (3*f1/2 + 3*(f2 + 2*f1))*math.exp(-t/6)
    )

  }


  val (test_data, test_targets) = batch(input_dim, domain._1, domain._2, gridSize = 10, ground_truth)

  val input = (FLOAT64, Shape(input_dim))

  val output = (Seq.fill(output_dim)(FLOAT64), Seq.fill(output_dim)(Shape(1)))


  val getAct = (s: Int) => (i: Int) =>
    if(i - s == 0) layer_trig(s"Act_$i")
    else dtflearn.Phi(s"Act_$i")

  val layer_sizes = (
    sizes :+ 1,
    sizes :+ 1
  )

  val (xs, ys) = (
    dtflearn.feedforward_stack(getAct(1), FLOAT64)(
      layer_sizes._1, 1,
      true),
    dtflearn.feedforward_stack(getAct(layer_sizes._1.length + 1), FLOAT64)(
      layer_sizes._2, layer_sizes._1.length + 1,
      true)
  )

  val (xs_params, ys_params) = (
    dtfutils.get_ffstack_properties(Seq(input_dim) ++ layer_sizes._1, 1, "FLOAT64"),
    dtfutils.get_ffstack_properties(Seq(input_dim) ++ layer_sizes._2, layer_sizes._1.length + 1, "FLOAT64")
  )


  val l2_reg = L2Regularization(
    xs_params._2.tail ++ ys_params._2.tail,
    xs_params._3.tail ++ ys_params._3.tail,
    xs_params._1.tail ++ ys_params._1.tail,
    reg = reg_param)

  val (x, y) = (
    q("xs", xs, false),
    q("ys", ys, false)
  )

  val (α, β, γ, δ) = (
    constant[Output]("alpha", Tensor(2.0/3)),
    constant[Output]("beta", Tensor(4.0/3)),
    constant[Output]("gamma", Tensor(1.0)),
    constant[Output]("delta", Tensor(1.0))
  )



  val lotka_volterra = Seq(
    ∇ - α*x + β*x*y,
    ∇ - δ*x*y + γ*y
  )

  val training_data = Seq(f1, f2).map(f => {
    dtfdata.supervised_dataset(
      Iterable(
        (
          dtf.tensor_f64(1)(0.0),
          dtf.tensor_f64(1)(f)
        )
      )
    )
  })

  val quadrature_scheme = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

  val nodes_tensor: Tensor = dtf.tensor_f64(
    quadrature_scheme.nodes.length, 1)(
    quadrature_scheme.nodes:_*
  )

  val weights_tensor: Tensor = dtf.tensor_f64(
    quadrature_scheme.nodes.length)(
    quadrature_scheme.weights:_*
  )

  val lorenz_system = dtflearn.dynamical_system[Double](
    Map("x" -> xs, "y" -> ys),
    lotka_volterra, input, output,
    tf.learn.L2Loss("Loss/L2") >> l2_reg >> tf.learn.Mean("L2/Mean"),
    nodes_tensor, weights_tensor, Tensor(1.0).reshape(Shape()))


  implicit val seqArgType: Function.ArgType[Seq[Output]] = seqFuncArgTypeConstructor(output_dim)

  val predator_prey_model = lorenz_system.solve(
    training_data,
    dtflearn.model.trainConfig(
      summary_dir,
      optimizer,
      dtflearn.abs_loss_change_stop(0.0001, maxIt),
      Some(
        dtflearn.model._train_hooks(
          summary_dir, stepRateFreq = 1000,
          summarySaveFreq = 1000,
          checkPointFreq = 1000)
      )),
    dtflearn.model.data_ops(training_data.size, training_data.size, 10)
  )

  val predictions = predator_prey_model.predict("x", "y")(test_data)

  //val error_tensor = predictions.subtract(test_targets)

  //val mae = error_tensor.abs.mean().scalar.asInstanceOf[Double]

  session.close()

  //print("Test Error is = ")
  //pprint.pprintln(mae)

  plot_outputs(test_data, predictions, "Predicted Populations")

  //plot_outputs(test_data, test_targets, "Actual Targets")



  (lorenz_system, predator_prey_model, training_data, plot3d_outputs(test_data, predictions))

}
