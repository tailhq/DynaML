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
import org.platanios.tensorflow.api.implicits.helpers.OutputToTensor
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}

case class AbstractDataSet[TI, OI, DI, SI, TT, OT, DT, ST](
  trainData: TI, trainLabels: TT, nTrain: Int,
  testData: TI, testLabels: TT, nTest: Int) {

  def training_data(
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

  def test_data(
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