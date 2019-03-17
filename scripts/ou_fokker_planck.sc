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
  min: Double,
  max: Double,
  gridSize: Int,
  func: Seq[Double] => Float): (Tensor[Float], Tensor[Float]) = {

  val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize) :+ max))

  val targets = points.map(func)

  (
    dtf.tensor_from[Float](Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten.map(_.toFloat)),
    dtf.tensor_from[Float](Seq.fill(dim)(gridSize + 1).product, 1)(targets)
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
def apply(num_data: Int = 100, optimizer: Optimizer = tf.train.AdaDelta(0.01f)) = {

  val session = Session()


  val tempdir = home/"tmp"

  val summary_dir = tempdir/s"dtf_wave1d_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

  val domain = (0.0, 5.0)

  val domain_size = domain._2 - domain._1

  val input_dim: Int = 2

  val output_dim: Int = 1


  val diff = 6.0
  val x0   = domain._2
  val th   = 3.0

  val ground_truth = (tl: Seq[Double]) => {
    val (t, x) = (tl.head, tl.last)

    (math.sqrt(th/(2*math.Pi*diff*(1 - math.exp(-2*th*t))))*
      math.exp(-1*th*math.pow(x - x0*math.exp(-th*t), 2)/(2*diff*(1 - math.exp(-2*th*t))))).toFloat
  }

  val f1 = (l: Double) => if(l == x0) 1d else 0d
  val f2 = (l: Double) => math.cos(2*math.Pi*l/domain_size)


  val (test_data, test_targets) = batch(input_dim, domain._1, domain._2, gridSize = 10, ground_truth)

  val input = Shape(2)

  val output = Shape(1)


  val function  =
    dtflearn.feedforward[Float](num_units = 5, useBias = true)(id = 1) >>
      tf.learn.Sigmoid("Act_1") >>
      dtflearn.feedforward[Float](num_units = 1, useBias = false)(id = 2)


  val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2)


  val rv = UniformRV(domain._1, domain._2)

  val training_data =
    dtfdata.supervised_dataset[Tensor[Float], Tensor[Float]](
      data = xs.flatMap(x => {

        val rand_time = rv.draw.toFloat

        Seq(
          (dtf.tensor_f32(1, input_dim)(0f, x.toFloat), dtf.tensor_f32(1, output_dim)(f1(x).toFloat)),
          (dtf.tensor_f32(1, input_dim)(rand_time, x.toFloat), dtf.tensor_f32(1, output_dim)(ground_truth(Seq(rand_time, x))))
        )
      }))


  val theta = variable[Output[Float], Float]("theta", Shape(), tf.OnesInitializer)
  val D     = variable[Output[Float], Float]("D", Shape(),     tf.OnesInitializer)
  val x     = q[Output[Float], Float](
    name = "X",
    dtflearn.layer(name = "X", MetaPipe[Mode, Output[Float], Output[Float]](m => xt => xt(---, 1::))),
    isSystemVariable = false)


  val ornstein_ulhenbeck = d_t - theta*d_s(x) - D*d_s(d_s)

  val analysis.GaussianQuadrature(nodes, weights) = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

  val nodes_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length*nodes.length, 2)(
    utils.combine(Seq(nodes.map(_.toFloat), nodes.map(_.toFloat))).flatten:_*
  )

  val weights_tensor: Tensor[Float] = dtf.tensor_f32(
    nodes.length*nodes.length)(
    utils.combine(Seq(weights.map(_.toFloat), weights.map(_.toFloat))).map(_.product):_*
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
      dtflearn.abs_loss_change_stop(0.001, 20000),
      Some(
        dtflearn.model._train_hooks(
          summary_dir, stepRateFreq = 5000,
          summarySaveFreq = 5000,
          checkPointFreq = 5000)
      )),
    dtflearn.model.data_ops(training_data.size/10, training_data.size, 10),
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
