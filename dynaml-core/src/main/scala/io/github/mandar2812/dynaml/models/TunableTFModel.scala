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
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, MetaPipe}
import io.github.mandar2812.dynaml.tensorflow.data.{DataSet, TFDataSet}
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.{Function, Output}

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
private[dynaml] class TunableTFModel[
IT, IO, IDA, ID, IS, I,
TT, TO, TDA, TD, TS, T](
  modelFunction: TunableTFModel.ModelFunc[
    IT, IO, IDA, ID, IS, I,
    TT, TO, TDA, TD, TS, T],
  hyp_params: Seq[String],
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

    val TFDataSet(train_split, validation_split) = _data_splits

    val (validation_inputs, validation_targets) = (
      validation_split.map((c: (IT, TT)) => c._1),
      validation_split.map((c: (IT, TT)) => c._2)
    )

    val model_instance = modelFunction(h)(train_split)

    model_instance.train()

    try {
      val predictions = model_instance.infer_coll(validation_inputs).map((c: (IT, TT)) => c._2)

      fitness_function(predictions.zip(validation_targets))
    } catch {
      case _: java.lang.IllegalStateException => Double.NaN
      case _: Throwable => Double.NaN
    }
  }
}

object TunableTFModel {

  type HyperParams = Map[String, Double]

  type ModelFunc[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T] = MetaPipe[
    HyperParams,
    DataSet[(IT, TT)],
    TFModel[
      IT, IO, IDA, ID, IS, I,
      TT, TO, TDA, TD, TS, T]
    ]

  /**
    * Create a fitness function using the Map-Reduce paradigm.
    *
    * */
  def map_reduce_fitness[TT](
    map_func: DataPipe[(TT, TT), Double],
    reduce_func: DataPipe2[Double, Double, Double]): DataPipe[DataSet[(TT, TT)], Double] =
    DataPipe((d: DataSet[(TT, TT)]) => d.map(map_func).reduce(reduce_func))


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
    data_handles: Option[(Input[IT, IO, IDA, ID, IS], Input[TT, TO, TDA, TD, TS])] = None)(
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

    val modelFunc = MetaPipe((h: TunableTFModel.HyperParams) => (data: DataSet[(IT, TT)]) => TFModel(
      data, architecture, input, target, processTarget, loss_func_gen(h), trainConfig,
      data_processing, inMemory, existingGraph, data_handles
    ))


    new TunableTFModel[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      modelFunc, hyp, training_data, fitness_function,
      validation_data, data_split_func
    )

  }



}