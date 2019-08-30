import ammonite.ops._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.models._
import io.github.mandar2812.dynaml.tensorflow.layers.L2Regularization
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import _root_.spire.implicits._
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.analysis._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe
import breeze.numerics.sigmoid

import scala.util.Random

val tempdir = home / "tmp"

val tf_summary_dir = tempdir / s"dtf_model_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"

val (weight, bias) = (2.5, 1.5)

val data_size = 5000
val rv        = GaussianRV(0.0, 2.0).iid(data_size)
val noise     = GaussianRV(0.0, 1.5)

val data = dtfdata
  .dataset(rv.draw)
  .to_supervised(
    DataPipe[Double, (Tensor[Double], Tensor[Double])](
      n =>
        (
          dtf.tensor_f64(1, 1)(n),
          dtf.tensor_f64(1)(n * weight + bias + noise.draw)
        )
    )
  )

val train_fraction = 0.7

val tf_dataset = data.partition(
  DataPipe[(Tensor[Double], Tensor[Double]), Boolean](
    _ => Random.nextDouble() <= train_fraction
  )
)

val architecture = dtflearn.feedforward[Double](num_units = 1)(id = 1)

val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
  dtfutils.get_ffstack_properties(d = 1, num_pred_dims = 1, Seq())

val scope = dtfutils.get_scope(architecture) _

val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

val loss_func_generator = (h: Map[String, Double]) => {
  tf.learn.L2Loss[Double, Double]("Loss/L2") >>
    tf.learn.Mean("Loss/Mean") >>
    L2Regularization[Double](
      layer_scopes,
      layer_parameter_names,
      layer_datatypes,
      layer_shapes,
      h("reg")
    ) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")
}

val tuning_config_generator =
  dtflearn.tunable_tf_model.ModelFunction.hyper_params_to_dir >>
    DataPipe(
      (p: Path) =>
        dtflearn.model
          .trainConfig[(Output[Double], Output[Double])](
            p,
            dtflearn.model.data_ops(
              10,
              1000,
              10,
              data_size / 5
            ),
            tf.train.Adam(0.1f),
            dtflearn.rel_loss_change_stop(0.005, 5000),
            Some(dtflearn.model._train_hooks(p))
          )
    )

val hyper_parameters = List(
  "reg"
)

val hyper_prior = Map(
  "reg" -> UniformRV(1e-4, math.pow(10, -2.5))
)

val fitness_function = DataPipe2[Output[Double], Output[Double], Output[Float]](
  (p, t) => p.subtract(t).square.sum(axes = -1).castTo[Float]
)

val convert_to_tensor =
  DynaMLPipe.identityPipe[(Tensor[Double], Tensor[Double])]

val pattern_to_tensor =
  dtfpipe.EagerStack[Double]().zip(dtfpipe.EagerStack[Double]())

val tunableTFModel: TunableTFModel[(Tensor[Double], Tensor[Double]), Output[
  Double
], Output[Double], Output[Double], Double, Tensor[Double], FLOAT64, Shape, Tensor[
  Double
], FLOAT64, Shape, Tensor[Double], FLOAT64, Shape] =
  dtflearn.tunable_tf_model(
    loss_func_generator,
    hyper_parameters,
    tf_dataset.training_dataset,
    dtflearn.model.tf_data_handle_ops(
      bufferSize = 500,
      patternToTensor = Some(pattern_to_tensor),
      concatOpO = Some(dtfpipe.EagerConcatenate[Double]()),
      caching_mode =
        dtflearn.model.data.FileCache(tf_summary_dir / "data_cache")
    ),
    Seq(fitness_function),
    architecture,
    (FLOAT64, Shape(1, 1)),
    (FLOAT64, Shape(1)),
    tuning_config_generator(tf_summary_dir),
    data_split_func = Some(
      DataPipe[(Tensor[Double], Tensor[Double]), Boolean](
        _ => scala.util.Random.nextGaussian() <= 0.7
      )
    ),
    inMemory = false
  )

val hyp_scaling = hyper_prior.map(
  p =>
    (
      p._1,
      Encoder(
        (x: Double) => (x - p._2.min) / (p._2.max - p._2.min),
        (u: Double) => u * (p._2.max - p._2.min) + p._2.min
      )
    )
)

val logit =
  Encoder((x: Double) => math.log(x / (1d - x)), (x: Double) => sigmoid(x))

val hyp_mapping = Some(
  hyper_parameters
    .map(
      h => (h, hyp_scaling(h) > logit)
    )
    .toMap
)

val gs = new CoupledSimulatedAnnealing(tunableTFModel, Some(hyp_scaling))
  .setPrior(hyper_prior)
  .setNumSamples(2)
  .setMaxIterations(2)

println("--------------------------------------------------------------------")
println("Initiating model tuning")
println("--------------------------------------------------------------------")

val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

println("--------------------------------------------------------------------")
println("Model tuning complete")
println("Chosen configuration:")
pprint.pprintln(config)
println("--------------------------------------------------------------------")
