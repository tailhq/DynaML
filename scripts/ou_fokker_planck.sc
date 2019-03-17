import _root_.io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}
import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.graphics.plot3d._
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde.{source => q, _}
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import ammonite.ops.home
import org.joda.time.DateTime
import _root_.org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

import scala.util.Random

val random = new Random()

def batch(
  dim: Int,
  min: Seq[Double],
  max: Seq[Double],
  gridSize: Int,
  func: Seq[Double] => Float): (Tensor[Float], Tensor[Float]) = {

  val points = utils.combine(Seq.tabulate(dim)(i => utils.range(min(i), max(i), gridSize) :+ max(i)))

  val targets = points.map(func)

  (
    dtf.tensor_f32(Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten.map(_.toFloat):_*),
    dtf.tensor_f32(Seq.fill(dim)(gridSize + 1).product, 1)(targets:_*)
  )
}

val layer = new Layer[Output[Float], Output[Float]]("Sin") {
  override val layerType = "Sin"

  override def forwardWithoutContext(input: Output[Float])(implicit mode: Mode): Output[Float] =
    tf.sin(input)
}

def plot_field(x: Tensor[Float], t: Tensor[Float]): DelauneySurface = {

  val size = x.shape(0)

  val data = (0 until size).map(row => {

    val inputs = (x(row, 0).scalar.asInstanceOf[Double], x(row, 1).scalar.asInstanceOf[Double])
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
  iterations: Int = 50000) = {

  val session = Session()


  val tempdir = home/"tmp"

  val summary_dir = tempdir/s"dtf_fokker_planck_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (-5.0, 5.0)

  val time_domain = (0d, 10d)

  val input_dim: Int = 2

  val output_dim: Int = 1


  val diff = 1.5
  val x0   = domain._2 - 1.5d
  val th   = 0.25

  val ground_truth = (tl: Seq[Double]) => {
    val (t, x) = (tl.head, tl.last)

    (math.sqrt(th/(2*math.Pi*diff*(1 - math.exp(-2*th*t))))*
      math.exp(-1*th*math.pow(x - x0*math.exp(-th*t), 2)/(2*diff*(1 - math.exp(-2*th*t))))).toFloat
  }

  val f1 = (l: Double) => if(l == x0) 1d else 0d


  val (test_data, test_targets) = batch(
    input_dim,
    Seq(time_domain._1, domain._1),
    Seq(time_domain._2, domain._2),
    gridSize = 20,
    ground_truth)

  val input = Shape(2)

  val output = Shape(1)


  val function  = dtflearn.feedforward_stack[Float]((i: Int) => tf.learn.Sigmoid(s"Act_$i"))(num_neurons ++ Seq(1))


  val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2)


  val rv = UniformRV(time_domain._1, time_domain._2)

  val training_data =
    dtfdata.supervised_dataset[Tensor[Float], Tensor[Float]](
      data = xs.flatMap(x => {

        val rand_time = rv.draw.toFloat

        Seq(
          (dtf.tensor_f32(1, input_dim)(0f, x.toFloat), dtf.tensor_f32(1, output_dim)(f1(x).toFloat)),
          (dtf.tensor_f32(1, input_dim)(rand_time, x.toFloat), dtf.tensor_f32(1, output_dim)(ground_truth(Seq(rand_time, x))))
        )
      }))


  val x_layer = dtflearn.layer(name = "X", MetaPipe[Mode, Output[Float], Output[Float]](m => xt => xt(---, 1::)))

  val theta = variable[Output[Float], Float](name = "theta", Shape(), tf.RandomTruncatedNormalInitializer(0.01f))
  val D     = variable[Output[Float], Float](name = "D", Shape(),     tf.RandomTruncatedNormalInitializer(0.01f))
  val x     = q[Output[Float], Float](name = "X", x_layer, isSystemVariable = false)


  val ornstein_ulhenbeck = d_t - theta*d_s(x) - D*d_s(d_s)

  val analysis.GaussianQuadrature(nodes, weights) = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)
  val analysis.GaussianQuadrature(nodes_t, weights_t) = analysis.eightPointGaussLegendre.scale(time_domain._1, time_domain._2)

  val nodes_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length*nodes_t.length, 2)(
    utils.combine(Seq(nodes_t.map(_.toFloat), nodes.map(_.toFloat))).flatten:_*
  )

  val weights_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length*nodes_t.length)(
    utils.combine(Seq(weights_t.map(_.toFloat), weights.map(_.toFloat))).map(_.product):_*
  )

  val wave_system1d = dtflearn.pde_system[Float, Float, Float](
    function,
    ornstein_ulhenbeck, input, output,
    tf.learn.L2Loss[Float, Float]("Loss/L2") >> tf.learn.Mean[Float]("L2/Mean"),
    nodes_tensor, weights_tensor, Tensor(1.0f).reshape(Shape()))


  val stackOperation = DataPipe[Iterable[Tensor[Float]], Tensor[Float]](bat => tfi.concatenate(bat.toSeq, axis = 0))

  val wave_model1d = wave_system1d.solve(
    training_data,
    dtflearn.model.trainConfig(
      summary_dir,
      optimizer,
      dtflearn.abs_loss_change_stop(0.001, iterations),
      Some(dtflearn.model._train_hooks(summary_dir))),
    dtflearn.model.data_ops(training_data.size/10, training_data.size/4, 10),
    concatOpI = Some(stackOperation),
    concatOpT = Some(stackOperation)
  )

  print("Test Data Shapes: ")
  pprint.pprintln(test_data.shape)
  pprint.pprintln(test_targets.shape)

  val predictions = wave_model1d.predict("Output")(test_data).head

  val plot = plot_field(test_data, predictions)

  val error_tensor = tfi.subtract(predictions, test_targets)

  val mae = tfi.mean(tfi.abs(error_tensor)).scalar


  session.close()

  print("Test Error is = ")
  pprint.pprintln(mae)

  val error_plot = plot_field(test_data, error_tensor)

  (wave_system1d, wave_model1d, training_data, plot, error_plot, mae)

}
