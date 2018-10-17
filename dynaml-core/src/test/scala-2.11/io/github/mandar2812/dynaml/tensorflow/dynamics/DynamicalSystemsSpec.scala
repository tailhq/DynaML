package io.github.mandar2812.dynaml.tensorflow.dynamics

import org.scalatest.{FlatSpec, Matchers}
import _root_.io.github.mandar2812.dynaml.analysis
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde._
import _root_.org.platanios.tensorflow.api._
import _root_.org.platanios.tensorflow.api.learn.Mode
import _root_.org.platanios.tensorflow.api.learn.layers.Layer
import ammonite.ops.home
import org.joda.time.DateTime
import _root_.io.github.mandar2812.dynaml.tensorflow.implicits._
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.{DataType, MathDataType}

import scala.util.Random


class DynamicalSystemsSpec extends FlatSpec with Matchers {

  val random = new Random()

  def batch(
    dim: Int,
    min: Double,
    max: Double,
    gridSize: Int,
    func: Seq[Double] => Double): (Tensor[FLOAT64], Tensor[FLOAT64]) = {

    val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize) :+ max))

    val targets = points.map(func)

    (
      dtf.tensor_from(FLOAT64, Seq.fill(dim)(gridSize + 1).product, dim)(points.flatten),
      dtf.tensor_from(FLOAT64, Seq.fill(dim)(gridSize + 1).product, 1)(targets)
    )
  }

  val layer = new Layer[Output, Output]("Sin") {
    override val layerType = "Sin"

    override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output =
      input.sin
  }

  "Dynamical systems API" should " learn canonical solution to the wave equation" in {

    val session = Session()

    val num_data = 50

    val tempdir = home/"tmp"

    val summary_dir = tempdir/s"dtf_wave1d_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm-ss")}"

    val domain = (0.0, 5.0)

    val domain_size = domain._2 - domain._1

    val input_dim: Int = 2

    val output_dim: Int = 1


    val ground_truth = (tl: Seq[Double]) => math.sin(2*math.Pi*tl.head/domain_size + 2*math.Pi*tl.last/domain_size)

    val f1 = (l: Double) => math.sin(2*math.Pi*l/domain_size)
    val f2 = (l: Double) => math.cos(2*math.Pi*l/domain_size)


    val (test_data, test_targets) = batch(input_dim, domain._1, domain._2, gridSize = 10, ground_truth)

    val input = (FLOAT64, Shape(2))

    val output = (Seq(FLOAT64), Seq(Shape(1)))


    val function  =
      dtflearn.feedforward(
        num_units = 2, useBias = false,
        ConstantInitializer(Tensor(1.0)),
        ConstantInitializer(Tensor(1.0)))(1) >>
        layer >>
        dtflearn.feedforward(
          num_units = 1, useBias = false,
          ConstantInitializer(Tensor(1.0)),
          ConstantInitializer(Tensor(1.0)))(2)


    val xs = utils.range(domain._1, domain._2, num_data) ++ Seq(domain._2)


    val training_data = dtfdata.supervised_dataset[Tensor[DataType], Tensor[DataType]](
      data = xs.flatMap(x => Seq(
        (dtf.tensor_f64(input_dim)(0, x), dtf.tensor_f64(output_dim)(f1(x))),
        (dtf.tensor_f64(input_dim)(domain_size/4, x), dtf.tensor_f64(output_dim)(f2(x)))
    )))


    val velocity = constant[Output, FLOAT64]("velocity", Tensor(1.0))


    val wave_equation = d_t(d_t) - velocity*d_s(d_s)

    val analysis.GaussianQuadrature(nodes, weights) = analysis.eightPointGaussLegendre.scale(domain._1, domain._2)

    val nodes_tensor: Tensor[FLOAT64] = dtf.tensor_f64(
      nodes.length*nodes.length, 2)(
      utils.combine(Seq(nodes, nodes)).flatten:_*
    )

    val weights_tensor: Tensor[FLOAT64] = dtf.tensor_f64(
      nodes.length*nodes.length)(
      utils.combine(Seq(weights, weights)).map(_.product):_*
    )

    val wave_system1d = dtflearn.dynamical_system[DataType](
      Map("wave_displacement" -> function),
      Seq(wave_equation), input, output,
      tf.learn.L2Loss("Loss/L2") >> tf.learn.Mean("L2/Mean"),
      nodes_tensor, weights_tensor, Tensor(1.0))


    implicit val seqArgType: Function.ArgType[Seq[Output]] = seqFuncArgTypeConstructor(1)

    val wave_model1d = wave_system1d.solve(
      Seq(training_data),
      dtflearn.model.trainConfig(
        summary_dir,
        tf.train.Adam(0.001f),
        dtflearn.abs_loss_change_stop(0.0001, 5000),
        Some(
          dtflearn.model._train_hooks(
            summary_dir, stepRateFreq = 1000,
            summarySaveFreq = 1000,
            checkPointFreq = 1000)
        )),
      dtflearn.model.data_ops(training_data.size/10, training_data.size, 10)
    )

    val predictions = wave_model1d.predict("wave_displacement")(test_data).head.cast(FLOAT64)

    val error_tensor = tfi.subtract(predictions, test_targets)

    val mae = tfi.mean(tfi.abs(error_tensor)).scalar

    session.close()

    val epsilon = 1E-3

    assert(mae.asInstanceOf[Double] <= epsilon)
  }



}
