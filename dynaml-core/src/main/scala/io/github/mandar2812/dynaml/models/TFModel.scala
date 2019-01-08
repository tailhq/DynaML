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
import org.platanios.tensorflow.api.ops.{Function, Output}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.ops.io.data.Data
import org.platanios.tensorflow.api.ops.io.data.Dataset

/**
  * <h4>Supervised Learning</h4>
  *
  * @tparam IT The type representing input tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam IO The type representing symbolic tensors of the input patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam IDA The underlying (scalar) data types of the input,
  *             e.g. `DataType.Aux[Double]`, `(DataType.Aux[Double], DataType.Aux[Double])` etc.
  * @tparam ID The input pattern's tensorflow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam IS The type of the input pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @tparam I The type of the symbolic tensor returned by the neural architecture,
  *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
  * @tparam ITT The type of the tensor returned by the neural architecture,
  *             corresponding to [[I]]
  * @tparam TT The type representing target/label tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam TO The type representing symbolic tensors of the target patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam TDA The underlying (scalar) data types of the targets,
  *             e.g. `DataType.Aux[Double]`, `(DataType.Aux[Double], DataType.Aux[Double])` etc.
  * @tparam TD The target pattern's tensorflow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam TS The type of the target pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @tparam T The type of the symbolic tensor of the processed targets, this is the type
  *           of the tensorflow symbol which is used to compute the loss.
  *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
  * @param architecture The network architecture,
  *                     takes a value of type [[IO]] and returns
  *                     a value of type [[I]].
  * @param input The input meta data.
  * @param target The output label meta data
  * @param processTarget A computation layer which converts
  *                      the original target of type [[TO]]
  *                      into a type [[T]], usable by the Estimator API
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
IT, IO, IDA, ID, IS, I, ITT,
TT, TO, TDA, TD, TS, T](
  override val g: DataSet[(IT, TT)],
  val architecture: Layer[IO, I],
  val input: (IDA, IS),
  val target: (TDA, TS),
  val processTarget: Layer[TO, T],
  val loss: Layer[(I, T), Output],
  val trainConfig: TFModel.TrainConfig,
  val data_processing: TFModel.DataOps = TFModel.data_ops(10000, 16, 10),
  val inMemory: Boolean = false,
  val existingGraph: Option[Graph] = None,
  data_handles: Option[TFModel.DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS]] = None,
  val concatOpI: Option[DataPipe[Iterable[IT], IT]] = None,
  val concatOpT: Option[DataPipe[Iterable[TT], TT]] = None)(
  implicit
  evDAToDI: DataTypeAuxToDataType.Aux[IDA, ID],
  evDToOI: DataTypeToOutput.Aux[ID, IO],
  evOToTI: OutputToTensor.Aux[IO, IT],
  evDataI: Data.Aux[IT, IO, ID, IS],
  evDAToDT: DataTypeAuxToDataType.Aux[TDA, TD],
  evDToOT: DataTypeToOutput.Aux[TD, TO],
  evOToTT: OutputToTensor.Aux[TO, TT],
  evDataT: Data.Aux[TT, TO, TD, TS],
  evDAToD: DataTypeAuxToDataType.Aux[(IDA, TDA), (ID, TD)],
  evData: Data.Aux[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
  evOToT: OutputToTensor.Aux[(IO, TO), (IT, TT)],
  evFunctionOutput: Function.ArgType[(IO, TO)],
  evFetchableIO: Fetchable.Aux[IO, IT],
  evFetchableI: Fetchable.Aux[I, ITT],
  evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ITT)],
  ev: Estimator.SupportedInferInput[IT, ITT, IT, IO, ID, IS, ITT]) extends
  Model[DataSet[(IT, TT)], IT, ITT] {



  if(data_processing.groupBuffer > 0)
    require(
      concatOpI.isDefined && concatOpT.isDefined,
      "`groupBuffer` is non zero but concatenate operation not defined. Set `concatOpI` and `concatOpT` variables")

  type ModelPair = dtflearn.SupModelPair[IT, IO, ID, IS, I, TT, TO, TD, TS, T]

  val concatOp: Option[DataPipe[Iterable[(IT, TT)], (IT, TT)]] =
    if(concatOpI.isEmpty || concatOpT.isEmpty) None
    else Some(DataPipe((it: Iterable[(IT, TT)]) => it.unzip) > concatOpI.get * concatOpT.get)

  private lazy val tf_dataset = TFModel._tf_data_set[(IT, TT), (IO, TO), (IDA, TDA), (ID, TD), (IS, TS)](
    g, data_processing,
    (input._1, target._1),
    (input._2, target._2),
    concatOp)

  private val TFModel.TrainConfig(summaryDir, optimizer, stopCriteria, trainHooks) = trainConfig

  lazy val (input_handle, target_handle): (Input[IT, IO, IDA, ID, IS], Input[TT, TO, TDA, TD, TS]) =
    if(data_handles.isDefined) data_handles.get
    else (
      tf.learn.Input[IT, IO, IDA, ID, IS](input._1, tf_dataset.outputShapes._1, "Input"),
      tf.learn.Input[TT, TO, TDA, TD, TS](target._1, tf_dataset.outputShapes._2, "Target"))

  private val graphInstance = if(existingGraph.isDefined) {
    println("Using existing provided TensorFlow graph")
    existingGraph.get
  } else Graph()

  val (model, estimator): ModelPair = tf.createWith(graph = graphInstance) {

    val m = tf.learn.Model.supervised(
      input_handle, architecture,
      target_handle, processTarget,
      loss, optimizer)

    val train_hooks = trainHooks match {
      case Some(hooks) => hooks
      case None => TFModel._train_hooks(summary_dir = summaryDir)
    }

    val config = tf.learn.Configuration(Some(summaryDir.toNIO))

    val e =
      if (inMemory) tf.learn.InMemoryEstimator(m, config, stopCriteria, train_hooks)
      else tf.learn.FileBasedEstimator(m, config, stopCriteria, train_hooks)

    (m, e)
  }

  def train(): Unit = estimator.train(() => tf_dataset)

  /**
    * @param point Input consisting of a nested structure of Tensors.
    * @return The model predictions of type [[TT]]
    * */
  override def predict(point: IT): ITT = estimator.infer[IT, ITT, ITT](() => point)

  /**
    * Generate predictions for a data set.
    *
    * Here `input_data` can be of one of the following types:
    *
    *   - A [[Dataset]], in which case this method returns an iterator over `(input, output)` tuples corresponding to
    *     each element in the dataset. Note that the predictions are computed lazily in this case, whenever an element
    *     is requested from the returned iterator.
    *   - A single input of type [[IT]], in which case this method returns a prediction of type [[I]].
    *
    * Note that, [[ModelInferenceOutput]] refers to the tensor type that corresponds to the symbolic type [[I]].
    * For example, if [[I]] is `(Output, Output)`, then [[ModelInferenceOutput]] will be `(Tensor, Tensor)`.
    *
    * */
  def infer[InferInput, InferOutput, ModelInferenceOutput](
    input_data: InferInput)(
    implicit
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): InferOutput = estimator.infer(() => input_data)

  /**
    * Generate predictions for a DynaML data set.
    *
    * @param input_data_set The data set containing input patterns
    * @return A DynaML data set of input-prediction tuples.
    * */
  def infer_coll(input_data_set: DataSet[IT])(
    implicit ev: Estimator.SupportedInferInput[
      Dataset[IT, IO, ID, IS],
      Iterator[(IT, ITT)],
      IT, IO, ID, IS, ITT],
    evFunctionOutput: org.platanios.tensorflow.api.ops.Function.ArgType[IO]
  ): Either[ITT, DataSet[ITT]] = concatOpI match {

    case None => Right(input_data_set.map((pattern: IT) => infer[IT, ITT, ITT](pattern)))

    case Some(concatFunc) => Left(infer[IT, ITT, ITT](concatFunc(input_data_set.data)))

  }

  /**
    * Close the underlying tensorflow graph.
    * */
  def close(): Unit = graphInstance.close()


}

object TFModel {

  type Handle[IT, IO, IDA, ID, IS]  = Input[IT, IO, IDA, ID, IS]

  type DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS] =
    (Handle[IT, IO, IDA, ID, IS], Handle[TT, TO, TDA, TD, TS])

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
    * @param groupBuffer If set to a value greater than 0,
    *                    use [[DataSet.build_buffered()]] instead
    *                    of the default [[DataSet.build()]].
    * */
  protected case class DataOps(
    shuffleBuffer: Int,
    batchSize: Int,
    prefetchSize: Int,
    groupBuffer: Int = 0)

  type Ops = DataOps

  /**
    * Generate a Tensorflow Dataset.
    *
    * @param data A DynaML data set consisting of nested Tensors.
    * @param ops A [[DataOps]] instance which specifies the
    *            TensorFlow Data API pipeline operations to perform.
    * @param data_type A nested structure of data types corresponding to the input tensors.
    * @param shape A nested structure of shapes corresponding to the inputs.
    * @param concatOp A data pipe which concatenates tensor groups.
    *                 Use only if [[DataOps.groupBuffer]] is set to a non zero value.
    *
    * */
  def _tf_data_set[T, O, DA, D, S](
    data: DataSet[T],
    ops: DataOps,
    data_type: DA,
    shape: S,
    concatOp: Option[DataPipe[Iterable[T], T]] = None)(
    implicit
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, O, D, S],
    evOToT: OutputToTensor.Aux[O, T],
    evFunctionOutput: Function.ArgType[O]): Dataset[T, O, D, S] = {

    val starting_data = if(ops.groupBuffer > 0) {
      require(
        concatOp.isDefined,
        "Concatenate Operation not defined. Set `concatOp` variable")

      data.build_buffered[T, O, DA, D, S](identityPipe[T], ops.groupBuffer, concatOp.get, data_type, shape).repeat()
    } else {

      data.build[T, O, DA, D, S](Left(identityPipe[T]), data_type, shape).repeat()
    }

    val shuffled_dataset = if(ops.shuffleBuffer > 0) starting_data.shuffle(ops.shuffleBuffer) else starting_data

    val batched_dataset = if(ops.batchSize > 0) shuffled_dataset.batch(ops.batchSize) else shuffled_dataset

    if(ops.prefetchSize > 0) batched_dataset.prefetch(ops.prefetchSize) else batched_dataset
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
  protected case class TrainConfig(
    summaryDir: Path,
    optimizer: Optimizer = tf.train.Adam(0.01),
    stopCriteria: StopCriteria = dtflearn.rel_loss_change_stop(0.05, 100000),
    trainHooks: Option[Set[Hook]] = None)

  type Config = TrainConfig

  /**
    * Creates a [[DataOps]] instance.
    * */
  val data_ops: DataOps.type        = DataOps

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
  IT, IO, IDA, ID, IS, I, ITT,
  TT, TO, TDA, TD, TS, T](
    g: DataSet[(IT, TT)],
    architecture: Layer[IO, I],
    input: (IDA, IS),
    target: (TDA, TS),
    processTarget: Layer[TO, T],
    loss: Layer[(I, T), Output],
    trainConfig: TFModel.TrainConfig,
    data_processing: TFModel.DataOps = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false,
    existingGraph: Option[Graph] = None,
    data_handles: Option[(Input[IT, IO, IDA, ID, IS], Input[TT, TO, TDA, TD, TS])] = None,
    concatOpI: Option[DataPipe[Iterable[IT], IT]] = None,
    concatOpT: Option[DataPipe[Iterable[TT], TT]] = None)(
    implicit
    evDAToDI: DataTypeAuxToDataType.Aux[IDA, ID],
    evDToOI: DataTypeToOutput.Aux[ID, IO],
    evOToTI: OutputToTensor.Aux[IO, IT],
    evDataI: Data.Aux[IT, IO, ID, IS],
    evDAToDT: DataTypeAuxToDataType.Aux[TDA, TD],
    evDToOT: DataTypeToOutput.Aux[TD, TO],
    evOToTT: OutputToTensor.Aux[TO, TT],
    evDataT: Data.Aux[TT, TO, TD, TS],
    evDAToD: DataTypeAuxToDataType.Aux[(IDA, TDA), (ID, TD)],
    evData: Data.Aux[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
    evOToT: OutputToTensor.Aux[(IO, TO), (IT, TT)],
    evFunctionOutput: Function.ArgType[(IO, TO)],
    evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ITT],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ITT)],
    ev: Estimator.SupportedInferInput[IT, ITT, IT, IO, ID, IS, ITT]) =
    new TFModel(
      g, architecture, input, target, processTarget, loss,
      trainConfig, data_processing, inMemory, existingGraph,
      data_handles, concatOpI, concatOpT
    )

}