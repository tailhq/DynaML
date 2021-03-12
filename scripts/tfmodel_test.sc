import ammonite.ops._
import io.github.tailhq.dynaml.pipes.DataPipe
import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.tensorflow._
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Layer

val tempdir = home / "tmp"

val summary_dir = tempdir / s"dtf_model_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"

val (weight, bias) = (2.5, 1.5)

val data_size = 1000
val rv        = GaussianRV(0.0, 2.0).iid(data_size)

val data = dtfdata
  .dataset(rv.draw)
  .to_supervised(
    DataPipe[Double, (Tensor[Double], Tensor[Double])](
      n => (dtf.tensor_f64(1)(n), dtf.tensor_f64(1)(n * weight + bias))
    )
  )

val test_data = dtfdata
  .dataset(rv.draw)
  .to_supervised(
    DataPipe[Double, (Tensor[Double], Tensor[Double])](
      n => (dtf.tensor_f64(1)(n), dtf.tensor_f64(1)(n * weight + bias))
    )
  )

val train_fraction = 0.7

val tf_dataset = data.partition(
  DataPipe[(Tensor[Double], Tensor[Double]), Boolean](
    _ => scala.util.Random.nextDouble() <= train_fraction
  )
)

val arch: Layer[Output[Double], Output[Double]] =
  dtflearn.feedforward[Double](num_units = 1)(id = 1)

val loss: Layer[(Output[Double], Output[Double]), Output[Double]] =
  tf.learn.L2Loss[Double, Double]("Loss/L2") >>
    tf.learn.Mean[Double]("Loss/Mean") >>
    tf.learn.ScalarSummary[Double]("Loss/ModelLoss", "ModelLoss")

val graphInstance = Graph()

val regression_model =
  dtflearn.model[Output[Double], Output[Double], Output[
    Double
  ], Double, Tensor[Double], FLOAT64, Shape, Tensor[Double], FLOAT64, Shape, Tensor[
    Double
  ], FLOAT64, Shape](
    arch,
    (FLOAT64, Shape(1)),
    (FLOAT64, Shape(1)),
    loss,
    existingGraph = Some(graphInstance)
  )

val batch_size = 10  
val num_epochs = 10L
val iterations_per_epoch = (data_size.toDouble / batch_size).toInt

val train_config = dtflearn.model.trainConfig(
  summary_dir,
  dtflearn.model.data_ops[(Output[Double], Output[Double])](
    shuffleBuffer = 5000,
    batchSize = batch_size,
    prefetchSize = 10
  ),
  tf.train.Adam(0.1f),
  dtflearn.rel_loss_change_stop(0.05, num_epochs*iterations_per_epoch, num_epochs),
  Some(
    dtflearn.model._train_hooks(
      summary_dir,
      stepRateFreq = iterations_per_epoch,
      summarySaveFreq = iterations_per_epoch,
      checkPointFreq = iterations_per_epoch
    )
  )
)

val pattern_to_tensor = DataPipe(
  (ds: Seq[(Tensor[Double], Tensor[Double])]) => {
    val (xs, ys) = ds.unzip

    (
      dtfpipe.EagerStack[Double](axis = 0).run(xs),
      dtfpipe.EagerStack[Double](axis = 0).run(ys)
    )
  }
)

val pattern_to_sym =
  DataPipe(
    (ds: Seq[(Tensor[Double], Tensor[Double])]) => {
      val (xs, ys)   = ds.unzip
      //val batch_size = ds.length
      val (xt, yt) = (
        dtfpipe.LazyStack[Double](axis = 0).run(xs.map(_.toOutput)),
        dtfpipe.LazyStack[Double](axis = 0).run(ys.map(_.toOutput))
      )

      (
        xt,
        yt
      )

    }
  )

regression_model.train(
  tf_dataset.training_dataset,
  train_config,
  dtflearn.model.tf_data_handle_ops(
    bufferSize = 500,
    //patternToSym = Some(pattern_to_sym),
    patternToTensor = Some(pattern_to_tensor),
    concatOpO = Some(dtfpipe.EagerConcatenate[Double]()),
    caching_mode = dtflearn.model.data.FileCache(summary_dir / "data_cache")
  )
)

val test_pred =
  regression_model.predict(Tensor[Double](1.0d).reshape(Shape(1, 1))).scalar

val metrics = regression_model.evaluate(
  test_data,
  Seq(
    dtflearn.mse[Output[Double], Double](),
    dtflearn.mae[Output[Double], Double]()
  ),
  dtflearn.model.data_ops[(Output[Double], Output[Double])](
    repeat = 0,
    shuffleBuffer = 0,
    batchSize = 16,
    prefetchSize = 10
  ),
  dtflearn.model.tf_data_handle_ops(
    bufferSize = 500,
    //patternToSym = Some(pattern_to_sym),
    patternToTensor = Some(pattern_to_tensor),
    concatOpO = Some(dtfpipe.EagerConcatenate[Double]()),
  )
)
