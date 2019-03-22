/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.models

import ammonite.ops.Path
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.tensorflow.data.DataSet
import io.github.mandar2812.dynaml.tensorflow.Learn
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.DynaMLPipe._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.data.Dataset

/**
  * <h4>Supervised Learning</h4>
  *
  * @tparam IT The type representing input tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam In The type representing symbolic tensors of the input patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam ID The input pattern's tensorflow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam IS The type of the input pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @tparam ArchOut The type of the symbolic tensor returned by the neural architecture,
  *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
  * @tparam TT The type representing target/label tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam Out The type representing symbolic tensors of the target patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam TD The target pattern's tensorflow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam TS The type of the target pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @param architecture The network architecture,
  *                     takes a value of type [[In]] and returns
  *                     a value of type [[ArchOut]].
  * @param input The input meta data.
  * @param target The output label meta data
  * @param loss The loss function to be optimized during training.
  *
  * @param trainConfig A [[_root_.io.github.mandar2812.dynaml.models.TFModel.TrainConfig]] instance, containing
  *                    information on how to carry out the learning process.
  *
  * @param data_processing The data processing operations to execute before launching model training.
  *                        See [[_root_.io.github.mandar2812.dynaml.models.TFModel.DataOps]]
  *
  * @param inMemory     Set to true if the estimator should be in-memory.
  * @author mandar2812 date 2018/09/11
  * */
class TFModel[
In, Out, ArchOut,
Loss: TF : IsFloatOrDouble,
IT, ID, IS,
TT, TD, TS,
ITT, IDD, ISS](
  val architecture: Layer[In, ArchOut],
  val input: (ID, IS),
  val target: (TD, TS),
  val loss: Layer[(ArchOut, Out), Output[Loss]],
  val inMemory: Boolean = false,
  val existingGraph: Option[Graph] = None,
  data_handles: Option[TFModel.DataHandles[In, Out]] = None)(
  implicit
  evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
  evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
  evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
  evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
  evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
  evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
  evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
  evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
  evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
  evOutputToShapeI: OutputToShape.Aux[In, IS],
  evOutputToShapeT: OutputToShape.Aux[Out, TS],
  evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
  evOutputStructure: OutputStructure[(In, Out)],
  evOutputStructureArchOut: OutputStructure[ArchOut],
  evOutputStructureI: OutputStructure[In],
  evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
  evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
  evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
  evOutputToTensorIn: OutputToTensor.Aux[In, IT],
  ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
  // This implicit helps the Scala 2.11 compiler.
  evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)])
  extends Predictor[IT, ITT] {

  type UnderlyingModel = (
    Option[dtflearn.SupervisedModel[In, Out, ArchOut, ArchOut, Loss]],
      Option[dtflearn.SupEstimatorTF[In, Out, ArchOut, ArchOut, Loss, (ArchOut, (In, Out))]])



  private val graphInstance = if(existingGraph.isDefined) {
    println("Using existing provided TensorFlow graph")
    existingGraph.get
  } else Graph()

  var (model, estimator): UnderlyingModel = (None, None)


  private def repeat(count: Int) = 
    if(count != 0) DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]](_.repeat(count))
    else identityPipe[Dataset[(In, Out)]]

  private def shuffle(buffer: Int, seed: Option[Int] = None) = 
    if(buffer > 0) DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]](_.shuffle(buffer, seed))
    else identityPipe[Dataset[(In, Out)]]

  private def shuffle_and_repeat(buffer: Int, count: Int, seed: Option[Int] = None) = 
    if(buffer > 0 &&  count != 0) DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]](_.shuffleAndRepeat(buffer, count, seed))
    else shuffle(buffer) > repeat(count)

  private def batch(batch_size: Int) = 
    if(batch_size > 0) DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]](_.batch[(ID, TD), (IS, TS)](batch_size))
    else identityPipe[Dataset[(In, Out)]]

  private def prefetch(buffer: Int) = 
    if(buffer > 0) DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]](_.prefetch(buffer))
    else identityPipe[Dataset[(In, Out)]]

  private def _get_tf_data(data: DataSet[(IT, TT)], data_ops: TFModel.Ops[IT, TT, ITT, In, Out]): Dataset[(In, Out)] = {

    val (concatOpI, concatOpT, concatOpO) = (data_ops.concatOpI, data_ops.concatOpT, data_ops.concatOpO)

    val process_pipe = data_ops.custom_data_pipe.getOrElse(
      TFModel.Data.shuffle_and_repeat[(In, Out)](data_ops.shuffleBuffer, data_ops.repeat, data_ops.seed) > 
      TFModel.Data.batch[(In, Out), (ID, TD), (IS, TS)](data_ops.batchSize) > 
      TFModel.Data.prefetch[(In, Out)](data_ops.prefetchSize)
    )

    val tf_dataset: Dataset[(In, Out)] = if(concatOpI.isDefined && concatOpT.isDefined) {

      val (concatI, concatT) = (concatOpI.get, concatOpT.get)

      val concatOp = DataPipe((batch: Iterable[(IT, TT)]) => {
        val (xs, ys) = batch.unzip
        (concatI(xs), concatT(ys))
      })

      data.build_buffered[(IT, TT), (In, Out), (ID, TD), (IS, TS)](
        data_ops.batchSize,
        identityPipe[(IT, TT)],
        concatOp,
        (input._1, target._1),
        (input._2, target._2))
    } else {
      data.build(
        identityPipe[(IT, TT)],
        (input._1, target._1),
        (input._2, target._2))
    }

    process_pipe(tf_dataset)
  }

  private def _get_tf_data_input(data: DataSet[IT], data_ops: TFModel.InputOps[IT, In])(
    implicit 
    evTensorToOutput: TensorToOutput.Aux[IT, In],
    evTensorToDataType: TensorToDataType.Aux[IT, ID],
    evTensorToShape: TensorToShape.Aux[IT, IS],
    evDataTypeToShape: DataTypeToShape.Aux[ID, IS]): Dataset[In] = {

    val concatOpI = data_ops.concatOpI

    val process_pipe = data_ops.custom_data_pipe.getOrElse(
      TFModel.Data.shuffle_and_repeat[In](data_ops.shuffleBuffer, data_ops.repeat, data_ops.seed) > 
      TFModel.Data.batch[In, ID, IS](data_ops.batchSize) > 
      TFModel.Data.prefetch[In](data_ops.prefetchSize)
    )

    val tf_dataset: Dataset[In] = if(concatOpI.isDefined) {

      val concatI = concatOpI.get

      data.build_buffered[IT, In, ID, IS](
        data_ops.batchSize,
        identityPipe[IT],
        concatI,
        input._1,
        input._2)
    } else {
      data.build[IT, In, ID, IS](
        identityPipe[IT],
        input._1,
        input._2)
    }

    process_pipe(tf_dataset)
  }

  def train(data: DataSet[(IT, TT)], trainConfig: TFModel.TrainConfig[IT, TT, ITT, In, Out]): Unit = {

    val TFModel.TrainConfig(summaryDir, data_ops, optimizer, stopCriteria, trainHooks) = trainConfig

    val tf_dataset: Dataset[(In, Out)] = _get_tf_data(data, data_ops)

    if(estimator.isEmpty) {

      val (input_handle, target_handle): TFModel.DataHandles[In, Out] =
        if(data_handles.isDefined) data_handles.get
        else (
          tf.learn.Input[In, ID, IS](input._1, tf_dataset.outputShapes._1, "Input"),
          tf.learn.Input[Out, TD, TS](target._1, tf_dataset.outputShapes._2, "Target"))

      val underlying_tf_pair = tf.createWith(graph = graphInstance) {

        val m = tf.learn.Model.simpleSupervised[In, Out, ArchOut, ArchOut, Loss](
          input_handle,
          target_handle,
          architecture,
          loss, optimizer)

        val train_hooks = trainHooks match {
          case Some(hooks) => hooks
          case None => if(inMemory) Set[Hook]() else TFModel._train_hooks(summary_dir = summaryDir)
        }

        val config = tf.learn.Configuration(Some(summaryDir.toNIO))

        val e =
          if (inMemory) tf.learn.InMemoryEstimator(m, config, stopCriteria, train_hooks)
          else tf.learn.FileBasedEstimator(m, config, stopCriteria, train_hooks)

        (Some(m), Some(e))
      }

      model     = underlying_tf_pair._1
      estimator = underlying_tf_pair._2
    }

    estimator.get.train[(ID, TD), (IS, TS)](() => tf_dataset)
  }

  protected def check_underlying_estimator(): Unit =
    require(
      estimator.isDefined,
      "Underlying TensorFlow Estimator is undefined! Either \\n " +
        "1. Model training probably threw an Exception or error. \\n " +
        "2. Model has been closed via close() method."
    )

  /**
    * Generate predictions for a data set.
    *
    * Here `input_data` can be of one of the following types:
    *
    *   - A [[Dataset]], in which case this method returns an iterator over `(input, output)` tuples corresponding to
    *     each element in the dataset. Note that the predictions are computed lazily in this case, whenever an element
    *     is requested from the returned iterator.
    *   - A single input of type [[IT]], in which case this method returns a prediction of type [[ITT]].
    *
    *
    * */
  override def predict(point: IT): ITT = {
    check_underlying_estimator()
    estimator.get.infer[IT, ID, IS, ITT, IDD, ISS, IT, ITT](() => point)
  }

  /**
    * Generate predictions for a data set.
    *
    * Here `input_data` can be of one of the following types:
    *
    *   - A [[Dataset]], in which case this method returns an iterator over `(input, output)` tuples corresponding to
    *     each element in the dataset. Note that the predictions are computed lazily in this case, whenever an element
    *     is requested from the returned iterator.
    *   - A single input of type [[IT]], in which case this method returns a prediction of type [[ITT]].
    *
    *
    * */
  def infer[InV, InD, InS, OutV, OutD, OutS, InferIn, InferOut](
    input_data: InferIn)(
    implicit
    //evOutputToDataTypeIn: OutputToDataType.Aux[In, InD],
    evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, OutD],
    //evOutputToShapeIn: OutputToShape.Aux[In, InS],
    evOutputToShapeOut: OutputToShape.Aux[ArchOut, OutS],
    evOutputToTensorIn: OutputToTensor.Aux[In, InV],
    evOutputToTensorOut: OutputToTensor.Aux[ArchOut, OutV],
    ev: Estimator.SupportedInferInput[In, InV, OutV, InferIn, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (InV, OutV)]): InferOut =
    estimator.get.infer(() => input_data)

  /**
    * Generate predictions for a DynaML data set.
    *
    * @param input_data_set The data set containing input patterns
    * @return A DynaML data set of predictions.
    * */
  def infer_coll(
    input_data_set: DataSet[IT],
    data_ops: TFModel.InputDataOps[IT, In] = TFModel.input_data_ops())(
    implicit 
    evTensorToOutput: TensorToOutput.Aux[IT, In],
    evTensorToDataType: TensorToDataType.Aux[IT, ID],
    evTensorToShape: TensorToShape.Aux[IT, IS],
    evDataTypeToShape: DataTypeToShape.Aux[ID, IS]): Iterator[(IT, ITT)] = {
    check_underlying_estimator()


    val tf_dataset: Dataset[In] = _get_tf_data_input(input_data_set, data_ops)

    infer(tf_dataset)
  }

  def infer_batch(
    input_data_set: DataSet[IT],
    data_ops: TFModel.Ops[IT, TT, ITT, In, Out] = TFModel.data_ops()): Either[ITT, DataSet[ITT]] = {

    val (concatOpI, concatOpO) = (data_ops.concatOpI, data_ops.concatOpO)

    check_underlying_estimator()

    val prediction_collection = concatOpI match {

      case None => input_data_set.map(DataPipe((pattern: IT) => infer(pattern)))

      case Some(concatFunc) => input_data_set
        .grouped(data_ops.batchSize)
        .map(DataPipe((batch: Seq[IT]) => concatFunc(batch)))
        .map(DataPipe((tensor_batch: IT) => infer(tensor_batch)))
    }

    concatOpO match {
      case None => Right(prediction_collection)
      case Some(concatOpFunc) => Left(concatOpFunc(prediction_collection.data))
    }

  }

  def evaluate(
    test_data: DataSet[(IT, TT)], 
    metrics: Seq[tf.metrics.Metric[(ArchOut, (In, Out)), Output[Float]]],
    evaluation_ops: TFModel.Ops[IT, TT, ITT, In, Out] = TFModel.data_ops(), 
    maxSteps: Long = -1L,
    saveSummaries: Boolean = true,
    name: String = null): Seq[Tensor[Float]] = {
      check_underlying_estimator()

      val (concatOpI, concatOpT, concatOpO) = (evaluation_ops.concatOpI, evaluation_ops.concatOpT, evaluation_ops.concatOpO)

      val tf_dataset: Dataset[(In, Out)] = _get_tf_data(test_data, evaluation_ops)

      val max_steps: Long = if(maxSteps < 0L) math.ceil(test_data.size.toDouble/evaluation_ops.batchSize).toLong else maxSteps

      estimator.get.evaluate(() => tf_dataset, metrics, max_steps, saveSummaries, name)
    }

  /**
    * Close the underlying tensorflow graph.
    * */
  def close(): Unit = {
    model = None
    estimator = None
    graphInstance.close()
  }


}

object TFModel {

  type Handle[IO]  = Input[IO]

  type DataHandles[IO, TO] = (Handle[IO], Handle[TO])

  /**
    * Defines data operations to be performed using TensorFlow data API.
    *
    * @param shuffleBuffer The size of the shuffle buffer, set to zero if
    *                      data shuffling is not needed.
    *
    * @param batchSize Size of the mini batch.
    *
    * @param prefetchSize Number of elements to prefetch.
    * 
    * @param repeat Decides how many times the tensorflow 
    *               data set is to be repeated. Defaults to -1
    *               which repeates it infinitely.
    * */
  protected case class DataOps[IT, TT, ITT, In, Out](
    shuffleBuffer: Int = 10000,
    batchSize: Int = 16,
    prefetchSize: Int = 10,
    repeat: Int = -1,
    seed: Option[Int] = None,
    concatOpI: Option[DataPipe[Iterable[IT], IT]] = None,
    concatOpT: Option[DataPipe[Iterable[TT], TT]] = None,
    concatOpO: Option[DataPipe[Iterable[ITT], ITT]] = None, 
    custom_data_pipe: Option[DataPipe[Dataset[(In, Out)], Dataset[(In, Out)]]] = None)

  type Ops[IT, TT, ITT, In, Out] = DataOps[IT, TT, ITT, In, Out]

  protected case class InputDataOps[IT, In](
    shuffleBuffer: Int = 10000,
    batchSize: Int = 16,
    prefetchSize: Int = 10,
    repeat: Int = -1,
    seed: Option[Int] = None,
    concatOpI: Option[DataPipe[Iterable[IT], IT]] = None, 
    custom_data_pipe: Option[DataPipe[Dataset[In], Dataset[In]]] = None)

  type InputOps[IT, In] = InputDataOps[IT, In]

  object Data {
    def repeat[T](count: Int) = 
    if(count != 0) DataPipe[Dataset[T], Dataset[T]](_.repeat(count))
    else identityPipe[Dataset[T]]

    def shuffle[T](buffer: Int, seed: Option[Int] = None) = 
    if(buffer > 0) DataPipe[Dataset[T], Dataset[T]](_.shuffle(buffer, seed))
    else identityPipe[Dataset[T]]

    def shuffle_and_repeat[T](buffer: Int, count: Int, seed: Option[Int] = None) = 
    if(buffer > 0 &&  count != 0) DataPipe[Dataset[T], Dataset[T]](_.shuffleAndRepeat(buffer, count, seed))
    else shuffle[T](buffer) > repeat[T](count)

    def batch[T, D, S](batch_size: Int)(implicit
    evOutputToDataType: OutputToDataType.Aux[T, D],
    evOutputToShape: OutputToShape.Aux[T, S]) = 
    if(batch_size > 0) DataPipe[Dataset[T], Dataset[T]](_.batch[D, S](batch_size))
    else identityPipe[Dataset[T]]

    def prefetch[T](buffer: Int) = 
    if(buffer > 0) DataPipe[Dataset[T], Dataset[T]](_.prefetch(buffer))
    else identityPipe[Dataset[T]]
  }


  /**
    * A training configuration, contains information on
    * optimization method, convergence test etc.
    *
    * @param optimizer The optimization algorithm implementation.
    * @param summaryDir A filesystem path of type [[ammonite.ops.Path]], which
    *                   determines where the intermediate model parameters/checkpoints
    *                   will be written.
    * @param stopCriteria The stopping criteria for training, for examples see
    *                     [[Learn.max_iter_stop]], [[Learn.abs_loss_change_stop]] and
    *                     [[Learn.rel_loss_change_stop]]
    * @param trainHooks   A set of training hooks to run, training hooks perform activities
    *                     such as saving the current TensorFlow graph, saving the loss, step rate,
    *                     summaries (audio, video, tensor) etc. See the `_train_hooks()` method
    *                     for setting some default training hooks.
    * */
  protected case class TrainConfig[IT, TT, ITT, In, Out](
    summaryDir: Path,
    data_processing: Ops[IT, TT, ITT, In, Out] = TFModel.data_ops(10000, 16, 10),
    optimizer: Optimizer = tf.train.Adam(0.01f),
    stopCriteria: StopCriteria = dtflearn.rel_loss_change_stop(0.05, 100000),
    trainHooks: Option[Set[Hook]] = None)

  type Config[IT, TT, ITT, In, Out] = TrainConfig[IT, TT, ITT, In, Out]

  /**
    * Creates a [[DataOps]] instance.
    * */
  val data_ops: DataOps.type        = DataOps

  val input_data_ops: InputDataOps.type = InputDataOps
  /**
    * Creates a [[TrainConfig]] instance.
    * */
  val trainConfig: TrainConfig.type = TrainConfig

  /**
    * Create a set of training hooks which save
    *
    * <ol>
    *   <li>The underlying TensorFlow graph</li>
    *   <li>Step Rate</li>
    *   <li>Summaries: Loss, Images and any others generated by the model layers/loss etc</li>
    * </ol>
    *
    * @param summary_dir The directory in which the model information is stored
    * @param stepRateFreq How frequently to write the step rate to disk
    * @param summarySaveFreq How frequently to save model summaries to disk
    * @param checkPointFreq How frequently to save model checkpoints.
    *
    * @return A collection of training hooks
    * */
  def _train_hooks(
    summary_dir: Path,
    stepRateFreq: Int = 5000,
    summarySaveFreq: Int = 5000,
    checkPointFreq: Int = 5000): Set[Hook] =
    Set(
      tf.learn.StepRateLogger(
        log = false, summaryDir = summary_dir.toNIO,
        trigger = tf.learn.StepHookTrigger(stepRateFreq)),
      tf.learn.SummarySaver(summary_dir.toNIO, tf.learn.StepHookTrigger(summarySaveFreq)),
      tf.learn.CheckpointSaver(summary_dir.toNIO, tf.learn.StepHookTrigger(checkPointFreq))
    )

  def apply[
  In, Out, ArchOut,
  Loss: TF : IsFloatOrDouble,
  IT, ID, IS,
  TT, TD, TS,
  ITT, IDD, ISS](
    architecture: Layer[In, ArchOut],
    input: (ID, IS),
    target: (TD, TS),
    loss: Layer[(ArchOut, Out), Output[Loss]],
    inMemory: Boolean = false,
    existingGraph: Option[Graph] = None,
    data_handles: Option[(Input[In], Input[Out])] = None)(
    implicit
    evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
    evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
    evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
    evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
    evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
    evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
    evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
    evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
    evOutputToShapeI: OutputToShape.Aux[In, IS],
    evOutputToShapeT: OutputToShape.Aux[Out, TS],
    evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
    evOutputStructure: OutputStructure[(In, Out)],
    evOutputStructureArchOut: OutputStructure[ArchOut],
    evOutputStructureI: OutputStructure[In],
    evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
    evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
    evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]) =
    new TFModel[
      In, Out, ArchOut, Loss,
      IT, ID, IS,
      TT, TD, TS,
      ITT, IDD, ISS](
      architecture, input, target, loss,
      inMemory, existingGraph,
      data_handles
    )

}