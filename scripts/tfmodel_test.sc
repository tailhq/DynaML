import ammonite.ops._
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.tensorflow._
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.types.DataType

import scala.util.Random


val tempdir = home/"tmp"

val summary_dir = tempdir/s"dtf_model_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"

val (weight, bias) = (2.5, 1.5)

val data_size = 100
val rv = GaussianRV(0.0, 2.0).iid(data_size)

val data = dtfdata.dataset(rv.draw).to_supervised(
  DataPipe[Double, (Tensor, Tensor)](n => (
    dtf.tensor_f64(1)(n),
    dtf.tensor_f64(1)(n*weight + bias))
  )
)

val train_fraction = 0.7

val tf_dataset = data.partition(
  DataPipe[(Tensor, Tensor), Boolean](_ => Random.nextDouble() <= train_fraction)
)

val arch = dtflearn.feedforward(1, true)(1)

val process_targets = dtflearn.identity[Output]("Id")

val loss = tf.learn.L2Loss("Loss/L2") >>
  tf.learn.Mean("Loss/Mean") >>
  tf.learn.ScalarSummary("Loss/ModelLoss", "ModelLoss")

val regression_model = dtflearn.model[
  Tensor, Output, DataType.Aux[Double], DataType, Shape, Output,
  Tensor, Output, DataType.Aux[Double], DataType, Shape, Output](
  tf_dataset.training_dataset,
  arch, (FLOAT64, Shape(1)), (FLOAT64, Shape(1)),
  process_targets, loss,
  dtflearn.model.trainConfig(
    summary_dir,
    tf.train.Adam(0.1),
    dtflearn.rel_loss_change_stop(0.05, 5000),
    Some(
      dtflearn.model._train_hooks(
        summary_dir, stepRateFreq = 1000,
        summarySaveFreq = 1000,
        checkPointFreq = 1000)
    )),
  dtflearn.model.data_ops(5000, 16, 10)
)



regression_model.train()
