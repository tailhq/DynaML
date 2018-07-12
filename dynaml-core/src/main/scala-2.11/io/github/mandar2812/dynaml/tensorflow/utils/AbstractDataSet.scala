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
package io.github.mandar2812.dynaml.tensorflow.utils

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.layers.Input
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}

case class AbstractDataSet[TI, TT](
  trainData: TI, trainLabels: TT, nTrain: Int,
  testData: TI, testLabels: TT, nTest: Int) {

  def training_data[OI, DI, SI, OT, DT, ST](
    implicit evData1: Data.Aux[TI,OI,DI,SI],
    evData2: Data.Aux[TT,OT,DT,ST],
    evOToT1: OutputToTensor.Aux[OI, TI],
    evOToT2: OutputToTensor.Aux[OT, TT],
    evFunctionInput1: Function.ArgType[OI],
    evFunctionInput2: Function.ArgType[OT])
  : Dataset[(TI, TT), (OI, OT), (DI, DT), (SI, ST)] =
    tf.data.TensorSlicesDataset[TI, OI, DI, SI](trainData).zip(
      tf.data.TensorSlicesDataset[TT, OT, DT, ST](trainLabels)
    )

  def test_data[OI, DI, SI, OT, DT, ST](
    implicit evData1: Data.Aux[TI,OI,DI,SI],
    evData2: Data.Aux[TT,OT,DT,ST],
    evOToT1: OutputToTensor.Aux[OI, TI],
    evOToT2: OutputToTensor.Aux[OT, TT],
    evFunctionInput1: Function.ArgType[OI],
    evFunctionInput2: Function.ArgType[OT])
  : Dataset[(TI, TT), (OI, OT), (DI, DT), (SI, ST)] =
    tf.data.TensorSlicesDataset[TI, OI, DI, SI](testData).zip(
      tf.data.TensorSlicesDataset[TT, OT, DT, ST](testLabels)
    )
}


/**
  * <h3>DynaML Tensorflow Data Set</h3>
  *
  * Wrapper class around the tensorflow [[Dataset]]
  * class, provides convenience functions for
  * boilerplate operations.
  * */
case class DataSet[T, O, D, S](
  data: Dataset[T, O, D, S],
  num: Long) {
  self =>

  def get: Dataset[T, O, D, S] = data

  def size: Long = num

  def handle[DA](_dataType: DA, name: String = "Input")(
    implicit evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evDToO: DataTypeToOutput.Aux[D, O],
    evOToT: OutputToTensor.Aux[O, T],
    evData: Data.Aux[T, O, D, S]): Input[T, O, DA, D, S] =
    tf.learn.Input[T, O, DA, D, S](_dataType, data.outputShapes, name)

  def zip[T2, O2, D2, S2](other_dataset: DataSet[T2, O2, D2, S2]): DataSet[(T, T2), (O, O2), (D, D2), (S, S2)] =
    DataSet(self.data.zip(other_dataset.data), self.size)

  def zip3[T2, O2, D2, S2, T3, O3, D3, S3](
    other1: DataSet[T2, O2, D2, S2],
    other2: DataSet[T3, O3, D3, S3]): DataSet[(T, T2, T3), (O, O2, O3), (D, D2, D3), (S, S2, S3)] =
    DataSet(self.data.zip3(other1.data, other2.data), self.size)

  def concatenate(others: DataSet[T, O, D, S]*): DataSet[T, O, D, S] = DataSet(
    others.map(_.data).foldLeft(self.data)((x, y) => x.concatenate(y)),
    self.size + others.map(_.size).sum
  )

  def batch(batchSize: Long, name: String = "Batch"): DataSet[T, O, D, S] =
    self.copy(data = self.data.batch(batchSize, name))

  def shuffle(
    bufferSize: Long,
    seed: Option[Int] = None,
    name: String = "Shuffle"): DataSet[T, O, D, S] =
    self.copy(data = self.data.shuffle(bufferSize, seed, name))

  def repeat(count: Long = -1, name: String = "Repeat"): DataSet[T, O, D, S] =
    self.copy(data = self.data.repeat(count, name))

  def prefetch(bufferSize: Long, name: String = "Prefetch"): DataSet[T, O, D, S] =
    self.copy(data = self.data.prefetch(bufferSize, name))

}


/**
  * <h3>Supervised Data Set</h3>
  *
  *
  * */
case class SupervisedDataSet[IT, IO, ID, IS, TT, TO, TD, TS](
  features: DataSet[IT, IO, ID, IS],
  targets: DataSet[TT, TO, TD, TS],
  num: Long) {

  self =>

  lazy val build: DataSet[(IT, TT), (IO, TO), (ID, TD), (IS, TS)] =
    features.zip(targets)

  def size: Long = num

  def input_handle[DA](_dataType: DA, name: String = "Input")(
    implicit evDAToD: DataTypeAuxToDataType.Aux[DA, ID],
    evDToO: DataTypeToOutput.Aux[ID, IO],
    evOToT: OutputToTensor.Aux[IO, IT],
    evData: Data.Aux[IT, IO, ID, IS]): Input[IT, IO, DA, ID, IS] =
    features.handle(_dataType, name)

  def output_handle[DA](_dataType: DA, name: String = "Output")(
    implicit evDAToD: DataTypeAuxToDataType.Aux[DA, TD],
    evDToO: DataTypeToOutput.Aux[TD, TO],
    evOToT: OutputToTensor.Aux[TO, TT],
    evData: Data.Aux[TT, TO, TD, TS]): Input[TT, TO, DA, TD, TS] =
    targets.handle(_dataType, name)

  /*
    def batch(batchSize: Long, name: String = "Batch"): SupervisedDataSet[IT, IO, ID, IS, TT, TO, TD, TS] =
      self.copy(features = self.features.batch(batchSize, name), targets = self.targets.batch(batchSize, name))
  */

}

object SupervisedDataSet {

  def apply[IT, IO, ID, IS, TT, TO, TD, TS](
    inputs: DataSet[IT, IO, ID, IS],
    outputs: DataSet[TT, TO, TD, TS]): SupervisedDataSet[IT, IO, ID, IS, TT, TO, TD, TS] = {

    require(inputs.size == outputs.size, "Inputs and Outputs must be of the same size!")

    SupervisedDataSet(inputs, outputs, inputs.size)

  }
}



/**
  *
  *
  * */
case class TFDataSet[IT, IO, ID, IS, TT, TO, TD, TS](
  training_data: SupervisedDataSet[IT, IO, ID, IS, TT, TO, TD, TS],
  test_data: SupervisedDataSet[IT, IO, ID, IS, TT, TO, TD, TS])