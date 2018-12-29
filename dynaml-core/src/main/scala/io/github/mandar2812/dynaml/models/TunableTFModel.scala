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

import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, MetaPipe}
import io.github.mandar2812.dynaml.tensorflow.data.{DataSet, TFDataSet}
import ammonite.ops._
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.{Function, Output}

import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}

/**
  * <h4>Hyper-parameter based Tensorflow Model</h4>
  *
  * @tparam IT The type representing input tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam IO The type representing symbolic tensors of the input patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam IDA The underlying (scalar) data types of the input,
  *             e.g. `DataType.Aux[Double]`, `(DataType.Aux[Double], DataType.Aux[Double])` etc.
  * @tparam ID The input pattern's TensorFlow data type,
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
  * @tparam TD The target pattern's TensorFlow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam TS The type of the target pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @tparam T The type of the symbolic tensor of the processed targets, this is the type
  *           of the TensorFlow symbol which is used to compute the loss.
  *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
  *
  * @param modelFunction A DynaML [[MetaPipe]] (Curried Function) which takes the
  *                      hyper-parameters as input, returns a [[DataPipe]] which
  *                      outputs a [[TFModel]] given some training data.
  *
  * @param hyp_params The string identifiers of the hyper-parameters.
  *
  * @param training_data Training data, as a DynaML [[DataSet]].
  *
  * @param fitness_function A [[DataPipe]] which takes a validation data collection
  *                         consisting of prediction-target tuples and outputs a
  *                         performance metric or fitness value. Ideally this value
  *                         should follow the "Lesser is better" paradigm.
  *
  * @param validation_data A validation data set, defaults to [[None]]. This data set
  *                        need not be specified if the [[data_split_func]] is defined.
  *
  * @param data_split_func A [[DataPipe]] which splits the [[training_data]] into train and
  *                        validation splits, defaults to [[None]]. This need not be specified
  *                        if [[validation_data]] is defined.
  *
  * */
class TunableTFModel[
IT, IO, IDA, ID, IS, I,
TT, TO, TDA, TD, TS, T](
  val modelFunction: TunableTFModel.ModelFunc[
    IT, IO, IDA, ID, IS, I,
    TT, TO, TDA, TD, TS, T],
  val hyp_params: Seq[String],
  protected val training_data: DataSet[(IT, TT)],
  val fitness_function: DataPipe[DataSet[(TT, TT)], Double],
  protected val validation_data: Option[DataSet[(IT, TT)]] = None,
  protected val data_split_func: Option[DataPipe[(IT, TT), Boolean]] = None)(
  implicit ev: Estimator.SupportedInferInput[
  Dataset[IT, IO, ID, IS],
  Iterator[(IT, TT)],
  IT, IO, ID, IS, TT],
  evFetchableI: Fetchable.Aux[I, TT],
  evFunctionOutput: org.platanios.tensorflow.api.ops.Function.ArgType[IO])
  extends GloballyOptimizable {

  implicit protected val formats: Formats = DefaultFormats

  override protected var hyper_parameters: List[String] = hyp_params.toList

  override protected var current_state: TunableTFModel.HyperParams = Map()

  protected def _data_splits: TFDataSet[(IT, TT)] = {

    require(
      validation_data.isDefined || data_split_func.isDefined,
      "If validation data is not explicitly provided, then data_split_func must be defined")

    if(validation_data.isEmpty) training_data.partition(data_split_func.get)
    else TFDataSet(training_data, validation_data.get)

  }

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: TunableTFModel.HyperParams, options: Map[String, String]): Double = {

    //Set the current state to `h`
    current_state = h

    //Obtain training and validation data splits
    val TFDataSet(train_split, validation_split) = _data_splits

    val (validation_inputs, validation_targets) = (
      validation_split.map((c: (IT, TT)) => c._1),
      validation_split.map((c: (IT, TT)) => c._2)
    )

    //Get the model instance.
    val model_instance = modelFunction(h)(train_split)
    //Train the model instance
    model_instance.train()

    //Compute the model fitness, guard against weird exceptions
    val fitness = try {
      val predictions = model_instance.infer_coll(validation_inputs).map((c: (IT, TT)) => c._2)

      fitness_function(predictions.zip(validation_targets))
    } catch {
      case _: java.lang.IllegalStateException => Double.NaN
      case _: Throwable => Double.NaN
    }

    //Append the model fitness to the hyper-parameter configuration
    val hyp_config_json = write_json(h + ("energy" -> fitness))

    //Write the configuration along with its fitness into the model
    //instance's summary directory
    write(
      model_instance.trainConfig.summaryDir/"state.csv",
      hyp_config_json)

    //Return the model fitness.
    fitness
  }
}

object TunableTFModel {

  type HyperParams = Map[String, Double]

  /**
    * Type-alias for "Model Functions".
    *
    * Model Functions take hyper-parameters as input
    * and return an instantiated TensorFlow Model [[TFModel]].
    *
    * */
  type ModelFunc[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = MetaPipe[
    HyperParams,
    DataSet[(IT, TT)],
    TFModel[
      IT, IO, IDA, ID, IS, I,
      TT, TO, TDA, TD, TS, T]
    ]

  /**
    * Type alias for "Fitness Functions"
    *
    * During hyper-parameter search, each model
    * instance is trained and evaluated on a validation
    * data set.
    *
    * After a model instance produces predictions for a
    * validation data set, fitness functions compute the
    * fitness from the predictions and data labels.
    * */
  type FitnessFunc[TT] = DataPipe[DataSet[(TT, TT)], Double]

  /**
    * <h4>Fitness Function Utility</h4>
    *
    * Provides methods setting up fitness function computations.
    * */
  object FitnessFunction {

    /**
      * Create a fitness function using the Map-Reduce paradigm.
      * This method assumes that the fitness function can be computed
      * by applying an element-wise `map` operation followed by a
      * `reduce` operation which computes the model's fitness function.
      *
      * @param map_func The element-wise map operation to apply to
      * */
    def apply[TT](
      map_func: DataPipe[(TT, TT), Double],
      reduce_func: DataPipe2[Double, Double, Double]): FitnessFunc[TT] =
      DataPipe((d: DataSet[(TT, TT)]) => d.map(map_func).reduce(reduce_func))

  }

  /**
    * <h4>Model Functions</h4>
    *
    * Helpful methods for creating [[ModelFunc]] instances, which
    * are needed for creating [[TunableTFModel]].
    * */
  object ModelFunction {


    val config_to_str: DataPipe[Map[String, Double], String] = DataPipe(_.map(c => s"${c._1}_${c._2}").mkString("-"))
    val generate_token: DataPipe[String, String]             = DataPipe(utils.tokenGenerator.generateMD5Token)

    private val to_token = config_to_str > generate_token

    /**
      * Create a [[ModelFunc]] from a "loss generator".
      *
      * @param loss_gen A function which takes the [[HyperParams]] and creates
      *                 the Loss function.
      *
      * @param architecture The model architecture.
      *
      * @param input Data type and shape of the model inputs.
      *
      * @param target Data type and shape of the model outputs/training labels.
      *
      * @param processTarget A layer which processes the labels/targets before using
      *                      them for training.
      *
      * @param trainConfig An instance of type [[TFModel.Config]], contains information
      *                    on the optimizer, summary directory and train hooks.
      *
      * @param data_processing An instance of type [[TFModel.DataOps]], contains details
      *                        on the data processing pipeline to be applied.
      *
      * @param inMemory Set to true if the model should be entirely in memory. Defaults
      *                 to false.
      *
      * @param existingGraph Defaults to None, set this parameter if the model should
      *                      be created in an existing TensorFlow graph.
      *
      * @param data_handles Defaults to None, set this parameter if you wish to instantiate
      *                     the model input-output handles.
      * */
    def from_loss_generator[
    IT, IO, IDA, ID, IS, I,
    TT, TO, TDA, TD, TS, T](
      loss_gen: HyperParams => Layer[(I, T), Output],
      architecture: Layer[IO, I],
      input: (IDA, IS),
      target: (TDA, TS),
      processTarget: Layer[TO, T],
      trainConfig: TFModel.Config,
      data_processing: TFModel.Ops,
      inMemory: Boolean = false,
      existingGraph: Option[Graph] = None,
      data_handles: Option[TFModel.DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS]] = None)(
      implicit evDAToDI: DataTypeAuxToDataType.Aux[IDA, ID],
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
      evFetchableI: Fetchable.Aux[I, IT],
      evFetchableIIO: Fetchable.Aux[(IO, I), (IT, IT)],
      ev: Estimator.SupportedInferInput[IT, TT, IT, IO, ID, IS, IT])
    : ModelFunc[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = {

      MetaPipe(
        (h: TunableTFModel.HyperParams) =>
          (data: DataSet[(IT, TT)]) => {


            val loss = loss_gen(h)

            val model_summaries = trainConfig.summaryDir/to_token(h)

            TFModel(
              data, architecture, input, target,
              processTarget, loss,
              trainConfig.copy(summaryDir = model_summaries),
              data_processing, inMemory, existingGraph,
              data_handles
            )
          }
      )
    }

    /**
      * Create a [[ModelFunc]] from a "architecture-loss generator".
      *
      * @param arch_loss_gen A function which takes the [[HyperParams]] and creates
      *                      an architecture-loss tuple.
      *
      * @param input Data type and shape of the model inputs.
      *
      * @param target Data type and shape of the model outputs/training labels.
      *
      * @param processTarget A layer which processes the labels/targets before using
      *                      them for training.
      *
      * @param trainConfig An instance of type [[TFModel.Config]], contains information
      *                    on the optimizer, summary directory and train hooks.
      *
      * @param data_processing An instance of type [[TFModel.DataOps]], contains details
      *                        on the data processing pipeline to be applied.
      *
      * @param inMemory Set to true if the model should be entirely in memory. Defaults
      *                 to false.
      *
      * @param existingGraph Defaults to None, set this parameter if the model should
      *                      be created in an existing TensorFlow graph.
      *
      * @param data_handles Defaults to None, set this parameter if you wish to instantiate
      *                     the model input-output handles.
      * */
    def from_arch_loss_generator[
    IT, IO, IDA, ID, IS, I,
    TT, TO, TDA, TD, TS, T](
      arch_loss_gen: HyperParams => (Layer[IO, I], Layer[(I, T), Output]),
      input: (IDA, IS),
      target: (TDA, TS),
      processTarget: Layer[TO, T],
      trainConfig: TFModel.Config,
      data_processing: TFModel.Ops,
      inMemory: Boolean = false,
      existingGraph: Option[Graph] = None,
      data_handles: Option[TFModel.DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS]] = None)(
      implicit evDAToDI: DataTypeAuxToDataType.Aux[IDA, ID],
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
      evFetchableI: Fetchable.Aux[I, IT],
      evFetchableIIO: Fetchable.Aux[(IO, I), (IT, IT)],
      ev: Estimator.SupportedInferInput[IT, TT, IT, IO, ID, IS, IT])
    : ModelFunc[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = {

      MetaPipe(
        (h: TunableTFModel.HyperParams) =>
          (data: DataSet[(IT, TT)]) => {

            val (architecture, loss) = arch_loss_gen(h)

            val model_summaries = trainConfig.summaryDir/to_token(h)

            TFModel(
              data, architecture, input, target,
              processTarget, loss,
              trainConfig.copy(summaryDir = model_summaries),
              data_processing, inMemory, existingGraph,
              data_handles
            )
          }
      )

    }
  }

  def apply[
  IT, IO, IDA, ID, IS, I,
  TT, TO, TDA, TD, TS, T](
    loss_func_gen: HyperParams => Layer[(I, T), Output],
    hyp: List[String],
    training_data: DataSet[(IT, TT)],
    fitness_function: DataPipe[DataSet[(TT, TT)], Double],
    architecture: Layer[IO, I],
    input: (IDA, IS),
    target: (TDA, TS),
    processTarget: Layer[TO, T],
    trainConfig: TFModel.Config,
    validation_data: Option[DataSet[(IT, TT)]] = None,
    data_split_func: Option[DataPipe[(IT, TT), Boolean]] = None,
    data_processing: TFModel.Ops = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false,
    existingGraph: Option[Graph] = None,
    data_handles: Option[TFModel.DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS]] = None)(
    implicit ev1: Estimator.SupportedInferInput[
    Dataset[IT, IO, ID, IS],
    Iterator[(IT, TT)],
    IT, IO, ID, IS, TT],
    evFetchableI1: Fetchable.Aux[I, TT],
    evFunctionOutput1: org.platanios.tensorflow.api.ops.Function.ArgType[IO],
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
    evFetchableI: Fetchable.Aux[I, IT],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, IT)],
    ev: Estimator.SupportedInferInput[IT, TT, IT, IO, ID, IS, IT])
  : TunableTFModel[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = {

    val modelFunc = ModelFunction.from_loss_generator(
      loss_func_gen, architecture, input, target,
      processTarget, trainConfig,
      data_processing, inMemory,
      existingGraph, data_handles
    )

    new TunableTFModel[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      modelFunc, hyp, training_data, fitness_function,
      validation_data, data_split_func
    )

  }

  def apply[
  IT, IO, IDA, ID, IS, I,
  TT, TO, TDA, TD, TS, T](
    arch_loss_gen: HyperParams => (Layer[IO, I], Layer[(I, T), Output]),
    hyp: List[String],
    training_data: DataSet[(IT, TT)],
    fitness_function: DataPipe[DataSet[(TT, TT)], Double],
    input: (IDA, IS),
    target: (TDA, TS),
    processTarget: Layer[TO, T],
    trainConfig: TFModel.Config,
    validation_data: Option[DataSet[(IT, TT)]],
    data_split_func: Option[DataPipe[(IT, TT), Boolean]],
    data_processing: TFModel.Ops,
    inMemory: Boolean,
    existingGraph: Option[Graph],
    data_handles: Option[TFModel.DataHandles[IT, IO, IDA, ID, IS, TT, TO, TDA, TD, TS]])(
    implicit ev1: Estimator.SupportedInferInput[
    Dataset[IT, IO, ID, IS],
    Iterator[(IT, TT)],
    IT, IO, ID, IS, TT],
    evFetchableI1: Fetchable.Aux[I, TT],
    evFunctionOutput1: org.platanios.tensorflow.api.ops.Function.ArgType[IO],
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
    evFetchableI: Fetchable.Aux[I, IT],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, IT)],
    ev: Estimator.SupportedInferInput[IT, TT, IT, IO, ID, IS, IT])
  : TunableTFModel[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = {

    val modelFunc = ModelFunction.from_arch_loss_generator(
      arch_loss_gen, input, target,
      processTarget, trainConfig,
      data_processing, inMemory,
      existingGraph, data_handles
    )

    new TunableTFModel[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      modelFunc, hyp, training_data, fitness_function,
      validation_data, data_split_func
    )

  }

}