import ammonite.ops._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.data.DataSet
import io.github.mandar2812.dynaml.tensorflow.layers.L2Regularization
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.types.DataType
import _root_.spire.implicits._
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe
import _root_.io.github.mandar2812.dynaml.analysis._

import scala.util.Random


val tempdir = home/"tmp"

val tf_summary_dir = tempdir/s"dtf_model_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"

val (weight, bias) = (2.5, 1.5)

val data_size = 5000
val rv = GaussianRV(0.0, 2.0).iid(data_size)

val data = dtfdata.dataset(rv.draw).to_supervised(
  DataPipe[Double, (Tensor, Tensor)](n => (
    dtf.tensor_f64(1, 1)(n),
    dtf.tensor_f64(1)(n*weight + bias))
  )
)

val train_fraction = 0.7

val tf_dataset = data.partition(
  DataPipe[(Tensor, Tensor), Boolean](_ => Random.nextDouble() <= train_fraction)
)

val architecture = dtflearn.feedforward(num_units = 1)(id = 1)

val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
  dtfutils.get_ffstack_properties(d = 1, num_pred_dims = 1, Seq())

val process_targets = dtflearn.identity[Output](name = "Id")

val loss_func_generator = (h: Map[String, Double]) => {

  val lossFunc = tf.learn.L2Loss("Loss/L2") >>
    tf.learn.Mean("Loss/Mean")

  lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, h("reg")) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")
}

implicit val detImpl = DynaMLPipe.identityPipe[Double]

val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.exp(x)),
  DifferentiableMap(
    (x: Double) => math.log(x),
    (x: Double) => 1.0/x)
)

val h10: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.pow(10d, x)),
  DifferentiableMap(
    (x: Double) => math.log10(x),
    (x: Double) => 1.0/(x*math.log(10d)))
)

val g1 = GaussianRV(0.0, 0.75)

val g2 = GaussianRV(0.2, 0.75)

val lg_p = h -> g1
val lg_e = h -> g2

val lu_reg = h10 -> UniformRV(-4d, -2.5d)

val hyper_parameters = List(
  "reg"
)

val hyper_prior = Map(
  "reg" -> UniformRV(1E-4, math.pow(10, -2.5))
)

val fitness_function = DataPipe2[Tensor, Tensor, Double](
  (p, t) => p.subtract(t).square.sum(axes = -1).mean().scalar.asInstanceOf[Double]
)


val train_config_tuning = dtflearn.model.trainConfig(
  tf_summary_dir, tf.train.Adam(0.1),
  dtflearn.rel_loss_change_stop(0.005, 5000)
)

val tf_data_ops = dtflearn.model.data_ops(10, 1000, 10, data_size/5)

/*val stackOperation = DataPipe[Iterable[(Tensor, Tensor)], (Tensor, Tensor)](bat =>
  (tfi.concatenate(bat.map(_._1).toSeq, axis = 0), tfi.concatenate(bat.map(_._2).toSeq, axis = 0))
)*/

val stackOperationI = DataPipe[Iterable[Tensor], Tensor](bat => tfi.concatenate(bat.toSeq, axis = 0))


val tunableTFModel: TunableTFModel[
  Tensor, Output, DataType.Aux[Double], DataType, Shape, Output, Tensor,
  Tensor, Output, DataType.Aux[Double], DataType, Shape, Output] =
  dtflearn.tunable_tf_model(
    loss_func_generator, hyper_parameters,
    tf_dataset.training_dataset,
    fitness_function,
    architecture,
    (FLOAT64, Shape(1, 1)),
    (FLOAT64, Shape(1)),
    tf.learn.Cast("TrainInput", FLOAT64),
    train_config_tuning,
    data_split_func = Some(DataPipe[(Tensor, Tensor), Boolean](_ => scala.util.Random.nextGaussian() <= 0.7)),
    data_processing = tf_data_ops,
    inMemory = false,
    concatOpI = Some(stackOperationI),
    concatOpT = Some(stackOperationI)
  )

val gs = new GridSearch[tunableTFModel.type](tunableTFModel)

gs.setPrior(hyper_prior)

gs.setNumSamples(2)


println("--------------------------------------------------------------------")
println("Initiating model tuning")
println("--------------------------------------------------------------------")

val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

println("--------------------------------------------------------------------")
println("Model tuning complete")
println("Chosen configuration:")
pprint.pprintln(config)
println("--------------------------------------------------------------------")

//println("Training final model based on chosen configuration")

