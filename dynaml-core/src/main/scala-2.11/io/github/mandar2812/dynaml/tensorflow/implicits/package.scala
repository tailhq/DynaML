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
package io.github.mandar2812.dynaml.tensorflow

import io.github.mandar2812.dynaml.pipes._
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe
import org.platanios.tensorflow.api.implicits.helpers.OutputToTensor
import org.platanios.tensorflow.api.learn.estimators.Estimator.SupportedInferInput
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}


package object implicits {

  implicit def singleOutputInferInput[T, O, D, S, I](
    implicit
    evOToT: OutputToTensor.Aux[O, T],
    ev: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]): SupportedInferInput[O, I, T, O, D, S, I] =
    new SupportedInferInput[O, I, T, O, D, S, I] {
      override def toDataset(value: O): Dataset[T, O, D, S] = tf.data.OutputDataset[T, O, D, S](value)
      override def convertFetched(iterator: Iterator[(T, I)]): I = iterator.next()._2
    }

  implicit val tensorSplit: MetaPipe12[Tensor, Int, Int, Tensor] = MetaPipe12(
    (workingData: Tensor) => (index_start: Int, index_end: Int) => workingData(index_start::index_end + 1, ---)
  )

  implicit val tensorTup2Split: MetaPipe12[(Tensor, Tensor), Int, Int, (Tensor, Tensor)] = MetaPipe12(
    (workingData: (Tensor, Tensor)) => (index_start: Int, index_end: Int) => (
      workingData._1(index_start::index_end + 1, ---),
      workingData._2(index_start::index_end + 1, ---)
    )
  )

  implicit val outputSplit: MetaPipe12[Output, Int, Int, Output] = MetaPipe12(
    (workingData: Output) => (index_start: Int, index_end: Int) => workingData(index_start::index_end + 1, ---)
  )

  implicit val concatTensorSplits: DataPipe[Iterable[Tensor], Tensor] =
    DataPipe((ds: Iterable[Tensor]) => tfi.concatenate(ds.toSeq, axis = 0))

  implicit val concatTensorTup2Splits: DataPipe[Iterable[(Tensor, Tensor)], (Tensor, Tensor)] = DataPipe(
    (ds: Iterable[(Tensor, Tensor)]) => {

      val separated_splits = ds.unzip

      (
        tfi.concatenate(separated_splits._1.toSeq, axis = 0),
        tfi.concatenate(separated_splits._2.toSeq, axis = 0)
      )

    })

  implicit val concatOutputTup2Splits: DataPipe[Iterable[(Output, Output)], (Output, Output)] = DataPipe(
    (ds: Iterable[(Output, Output)]) => {

      val separated_splits = ds.unzip

      (
        tf.concatenate(separated_splits._1.toSeq).toOutput,
        tf.concatenate(separated_splits._2.toSeq).toOutput
      )

    })

  implicit val convOutputToOutput: DataPipe[Output, Output] = DynaMLPipe.identityPipe[Output]

  implicit val convTensorToOutput: DataPipe[Tensor, Output] = DataPipe((t: Tensor) => t.toOutput)

  implicit val convOutputTensorToOutputTup: DataPipe[(Output, Tensor), (Output, Output)] =
    convOutputToOutput * convTensorToOutput

  implicit val convTensorOutputToOutputTup: DataPipe[(Tensor, Output), (Output, Output)] =
    convTensorToOutput * convOutputToOutput

  implicit val convTensorTupToOutputTup: DataPipe[(Tensor, Tensor), (Output, Output)] =
    convTensorToOutput * convTensorToOutput

  implicit val convOutputTupToOutputTup: DataPipe[(Output, Output), (Output, Output)] =
    convOutputToOutput * convOutputToOutput




}
