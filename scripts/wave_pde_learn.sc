import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.DelauneySurface
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde._
import _root_.org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import ammonite.ops.home
import org.joda.time.DateTime
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer
import _root_.io.github.mandar2812.dynaml.tensorflow.implicits._
import org.platanios.tensorflow.api.ops.Function

import scala.util.Random


val random = new Random()

def batch(dim: Int, min: Double, max: Double, gridSize: Int): Tensor = {

  val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize)))


  dtf.tensor_f64(Seq.fill(dim)(gridSize).product, dim)(points.flatten:_*)
}

val layer = new Layer[Output, Output]("Sin") {
  override val layerType = "Sin"

  override protected def _forward(input: Output)(implicit mode: Mode): Output =
    input.sin
}

def plot_field(x: Tensor, t: Tensor): DelauneySurface = {

  val size = x.shape(0)

  val data = (0 until size).map(row => {

    val inputs = (x(row, 0).scalar.asInstanceOf[Double], x(row, 1).scalar.asInstanceOf[Double])
    val output = t(row).scalar.asInstanceOf[Double]

    (inputs, output)
  })

  plot3d.draw(data)
}


@main
def apply() = {


  val graphInstance = Graph()

  val tempdir = home/"tmp"

  val summary_dir = tempdir/s"dtf_wave1d_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"


  val domain = (0.0, 5.0)

  val domain_size = domain._2 - domain._1

  val input_dim: Int = 2

  val output_dim: Int = 1

  val input = (FLOAT64, Shape(2))

  val output = (Seq(FLOAT64), Seq(Shape(1)))


  tf.createWith(graph = graphInstance) {
    val function  =
      dtflearn.feedforward(2, false)(1) >>
        layer >>
        dtflearn.feedforward(1, true)(2)


    val xs = utils.range(domain._1, domain._2, 100) ++ Seq(domain._2)

    val f1 = (l: Double) => math.sin(2*math.Pi*l/domain_size)
    val f2 = (l: Double) => -math.cos(2*math.Pi*l/domain_size)


    val training_data = dtfdata.supervised_dataset(xs.flatMap(x => Seq(
      (dtf.tensor_f64(input_dim)(0, x), dtf.tensor_f64(output_dim)(f1(x))),
      (dtf.tensor_f64(input_dim)(domain_size/4, x), dtf.tensor_f64(output_dim)(f2(x)))
    )))


    val velocity  = constant[Output]("velocity", Tensor(1.0))


    val wave_equation   = d_t(d_t) - d_s(d_s)*velocity


    val analysis.GaussianQuadrature(nodes, weights) = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

    val nodes_tensor: Tensor = dtf.tensor_f64(
      nodes.length*nodes.length, 2)(
      utils.combine(Seq(nodes, nodes)).flatten:_*
    )

    val weights_tensor: Tensor = dtf.tensor_f64(
      nodes.length*nodes.length)(
      utils.combine(Seq(weights, weights)).map(_.product):_*
    )

    val total_quantities = 1 + wave_equation.variables.toSeq.length

    val data_handles = (
      tf.learn.Input(Seq(input._1), Seq(Shape(-1, 2)), "Input"),
      tf.learn.Input(output._1, Seq(Shape(-1, 1)), "Outputs"))

    val wave_system1d = dtflearn.dynamical_system[Double](
      Map("wave_displacement" -> function),
      Seq(wave_equation), input, output,
      tf.learn.L2Loss("Loss/L2") >> tf.learn.Mean("L2/Mean"),
      nodes_tensor, weights_tensor,
      Tensor(0.5), graphInstance)


    implicit val seqArgType: Function.ArgType[Seq[Output]] = seqFuncArgTypeConstructor(1)

    val wave_model1d = wave_system1d.solve(
      Seq(training_data),
      dtflearn.model.trainConfig(
        summary_dir,
        tf.train.Adam(0.01),
        dtflearn.rel_loss_change_stop(0.05, 5000),
        Some(
          dtflearn.model._train_hooks(
            summary_dir, stepRateFreq = 1000,
            summarySaveFreq = 1000,
            checkPointFreq = 1000)
        )),
      dtflearn.model.data_ops(training_data.size, training_data.size, 10)
    )

    val test_data = batch(2, domain._1, domain._2, 100)

    val predictions = wave_model1d.predict("wave_displacement")(test_data).head

    val plot = plot_field(test_data, predictions)

    (wave_system1d, wave_model1d, training_data, plot)
  }

}
