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
import org.platanios.tensorflow.api.implicits.helpers.{StructureFromDataType, StructureFromTensor}
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.ops.io.data.Data

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
private[dynaml] class TFModel[
IT, IO, ID, IS, I,
TT, TO, TD, TS, T,
InferInput, InferOutput,
ModelInferenceOutput](
  override val g: DataSet[(IT, TT)],
  val architecture: Layer[IO, I],
  val input: (ID, IS),
  val target: (TD, TS),
  val processTarget: Layer[TO, T],
  val loss: Layer[(I, T), Output],
  val trainConfig: TFModel.TrainConfig,
  val data_processing: TFModel.DataOps = TFModel.data_ops(10000, 16, 10),
  val inMemory: Boolean = false,
  val existingGraph: Option[Graph] = None,
  data_handles: Option[(Input[IT, IO, ID, IS], Input[TT, TO, TD, TS])] = None)(
  implicit
  evStructureFromTensor: StructureFromTensor.Aux[(IT, TT),(IO, TO),(ID, TD),(IS, TS)],
  evStructureI: StructureFromDataType.Aux[IT, IO, ID, IS],
  evStructureT: StructureFromDataType.Aux[TT, TO, TD, TS],
  evDataI: Data.Aux[IT, IO, ID, IS],
  evDataT: Data.Aux[TT, TO, TD, TS],
  evData: Data.Aux[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
  evFunctionOutput: Function.ArgType[(IO, TO)],
  evFetchableIO: Fetchable.Aux[IO, IT],
  evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
  evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
  ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]) extends
  Model[DataSet[(IT, TT)], InferInput, InferOutput] {

  type ModelPair = dtflearn.SupModelPair[IT, IO, ID, IS, I, TT, TO, TD, TS, T]

  private lazy val tf_dataset = g.build[(IT, TT), (IO, TO), (ID, TD), (IS, TS)](
    Left(identityPipe[(IT, TT)]),
    (input._1, target._1),
    (input._2, target._2))
    .repeat()
    .shuffle(data_processing.shuffleBuffer)
    .batch(data_processing.batchSize)
    .prefetch(data_processing.prefetchSize)

  private val TFModel.TrainConfig(summaryDir, optimizer, stopCriteria, trainHooks) = trainConfig

  lazy val (input_handle, target_handle): (Input[IT, IO, ID, IS], Input[TT, TO, TD, TS]) =
    if(data_handles.isDefined) data_handles.get
    else (
      tf.learn.Input[IT, IO, ID, IS](input._1, tf_dataset.outputShapes._1, "Input"),
      tf.learn.Input[TT, TO, TD, TS](target._1, tf_dataset.outputShapes._2, "Target"))

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
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: InferInput): InferOutput = estimator.infer(() => point)
}

object TFModel {

  protected case class DataOps(shuffleBuffer: Int, batchSize: Int, prefetchSize: Int)

  type Ops = DataOps

  /**
    * A training configuration, contains information on
    * optimization method, convergence test etc.
    *
    * @param optimizer The optimization algorithm implementation.
    * @param summaryDir A filesystem path of type [[ammonite.ops.Path]], which
    *                     determines where the intermediate model parameters/checkpoints
    *                     will be written.
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
    optimizer: Optimizer = tf.train.Adam(0.01f),
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
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, T,
  InferInput, InferOutput,
  ModelInferenceOutput](
  g: DataSet[(IT, TT)],
  architecture: Layer[IO, I],
  input: (ID, IS),
  target: (TD, TS),
  processTarget: Layer[TO, T],
  loss: Layer[(I, T), Output],
  trainConfig: TFModel.TrainConfig,
  data_processing: TFModel.DataOps = TFModel.data_ops(10000, 16, 10),
  inMemory: Boolean = false,
  existingGraph: Option[Graph] = None,
  data_handles: Option[(Input[IT, IO, ID, IS], Input[TT, TO, TD, TS])] = None)(
    implicit
    evStructureFromTensor: StructureFromTensor.Aux[(IT, TT),(IO, TO),(ID, TD),(IS, TS)],
    evStructureI: StructureFromDataType.Aux[IT, IO, ID, IS],
    evStructureT: StructureFromDataType.Aux[TT, TO, TD, TS],
    evDataI: Data.Aux[IT, IO, ID, IS],
    evDataT: Data.Aux[TT, TO, TD, TS],
    evData: Data.Aux[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
    evFunctionOutput: Function.ArgType[(IO, TO)],
    evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]) =
    new TFModel(
      g, architecture, input, target, processTarget, loss,
      trainConfig, data_processing, inMemory, existingGraph,
      data_handles
    )

}