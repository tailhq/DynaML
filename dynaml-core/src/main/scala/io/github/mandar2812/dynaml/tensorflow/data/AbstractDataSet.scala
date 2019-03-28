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
package io.github.mandar2812.dynaml.tensorflow.data

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.ops.data.Dataset

case class AbstractDataSet[TI, TT](
  trainData: TI, trainLabels: TT, nTrain: Int,
  testData: TI, testLabels: TT, nTest: Int) {

  def training_data[OI, DI, SI, OT, DT, ST](
    implicit
    evTensorToOutput: TensorToOutput.Aux[TI, OI],
    evTensorToDataType: TensorToDataType.Aux[TI, DI],
    evTensorToShape: TensorToShape.Aux[TI, SI],
    evOutputStructure: OutputStructure[OI],
    evTensorToOutputT: TensorToOutput.Aux[TT, OT],
    evTensorToDataTypeT: TensorToDataType.Aux[TT, DT],
    evTensorToShapeT: TensorToShape.Aux[TT, ST],
    evOutputStructureT: OutputStructure[OT],
    evOutputToDataTypeT: OutputToDataType.Aux[OI, DI],
    evOutputToShapeT: OutputToShape.Aux[OI, SI],
    evOutputToDataTypeR: OutputToDataType.Aux[OT, DT],
    evOutputToShapeR: OutputToShape.Aux[OT, ST]): Dataset[(OI, OT)] =
    tf.data
      .datasetFromTensorSlices[OI, TI, DI, SI](trainData)
      .zip[DI, SI, OT, DT, ST](tf.data.datasetFromTensorSlices[OT, TT, DT, ST](trainLabels))

  def test_data[OI, DI, SI, OT, DT, ST](
    implicit
    evTensorToOutput: TensorToOutput.Aux[TI, OI],
    evTensorToDataType: TensorToDataType.Aux[TI, DI],
    evTensorToShape: TensorToShape.Aux[TI, SI],
    evOutputStructure: OutputStructure[OI],
    evTensorToOutputT: TensorToOutput.Aux[TT, OT],
    evTensorToDataTypeT: TensorToDataType.Aux[TT, DT],
    evTensorToShapeT: TensorToShape.Aux[TT, ST],
    evOutputStructureT: OutputStructure[OT],
    evOutputToDataTypeT: OutputToDataType.Aux[OI, DI],
    evOutputToShapeT: OutputToShape.Aux[OI, SI],
    evOutputToDataTypeR: OutputToDataType.Aux[OT, DT],
    evOutputToShapeR: OutputToShape.Aux[OT, ST]): Dataset[(OI, OT)] =
    tf.data
      .datasetFromTensorSlices[OI, TI, DI, SI](testData)
      .zip(tf.data.datasetFromTensorSlices[OT, TT, DT, ST](testLabels))
}

