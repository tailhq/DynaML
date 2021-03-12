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
package io.github.tailhq.dynaml.tensorflow

import io.github.tailhq.dynaml.pipes._
import org.platanios.tensorflow.api.{DataType, _}
import _root_.io.github.tailhq.dynaml.DynaMLPipe
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.learn.estimators.Estimator.SupportedInferInput
import org.platanios.tensorflow.api.ops.Function




package object implicits {

  /*implicit def singleOutputInferInput[T, O, D, S, I](
    implicit
    evStructure: StructureFromOutput.Aux[T, O, D, S],
    evData: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]): SupportedInferInput[O, I, T, O, D, S, I] =
    new SupportedInferInput[O, I, T, O, D, S, I] {
      override def toDataset(value: O): Dataset[T, O, D, S] = tf.data.OutputDataset[T, O, D, S](value)
      override def convertFetched(iterator: Iterator[(T, I)]): I = iterator.next()._2
    }*/

  implicit def tensorSplit[D: TF]: MetaPipe12[Tensor[D], Int, Int, Tensor[D]] = MetaPipe12(
    (workingData: Tensor[D]) => (index_start: Int, index_end: Int) => workingData(index_start::index_end + 1, ---)
  )

  implicit def tensorTup2Split[D: TF]:
  MetaPipe12[(Tensor[D], Tensor[D]), Int, Int, (Tensor[D], Tensor[D])] = MetaPipe12(
    (workingData: (Tensor[D], Tensor[D])) => (index_start: Int, index_end: Int) => (
      workingData._1(index_start::index_end + 1, ---),
      workingData._2(index_start::index_end + 1, ---)
    )
  )

  implicit def outputSplit[D: TF]: MetaPipe12[Output[D], Int, Int, Output[D]] = MetaPipe12(
    (workingData: Output[D]) => (index_start: Int, index_end: Int) => workingData(index_start::index_end + 1, ---)
  )

  implicit def concatTensorSplits[D: TF]: DataPipe[Iterable[Tensor[D]], Tensor[D]] =
    DataPipe((ds: Iterable[Tensor[D]]) => dtf.concatenate(ds.toSeq, axis = 0))

  implicit def concatTensorTup2Splits[D: TF]:
  DataPipe[Iterable[(Tensor[D], Tensor[D])], (Tensor[D], Tensor[D])] =
    DataPipe((ds: Iterable[(Tensor[D], Tensor[D])]) => {

      val separated_splits: (Iterable[Tensor[D]], Iterable[Tensor[D]]) = ds.unzip


      (
        dtf.concatenate[D](separated_splits._1.toSeq, 0),
        dtf.concatenate[D](separated_splits._2.toSeq, 0)
      )

    })

  implicit def concatOutputTup2Splits[D: TF]: DataPipe[Iterable[(Output[D], Output[D])], (Output[D], Output[D])] =
    DataPipe((ds: Iterable[(Output[D], Output[D])]) => {

      val separated_splits = ds.unzip

      (
        tf.concatenate(separated_splits._1.toSeq),
        tf.concatenate(separated_splits._2.toSeq)
      )

    })

  implicit def convOutputToOutput[D: TF]: DataPipe[Output[D], Output[D]] = DynaMLPipe.identityPipe[Output[D]]

  implicit def convTensorToOutput[D: TF]: DataPipe[Tensor[D], Output[D]] = DataPipe((t: Tensor[D]) => t.toOutput)

  implicit def convOutputTensorToOutputTup[D: TF]: DataPipe[(Output[D], Tensor[D]), (Output[D], Output[D])] =
    convOutputToOutput[D] * convTensorToOutput[D]

  implicit def convTensorOutputToOutputTup[D: TF]: DataPipe[(Tensor[D], Output[D]), (Output[D], Output[D])] =
    convTensorToOutput[D] * convOutputToOutput[D]

  implicit def convTensorTupToOutputTup[D: TF]: DataPipe[(Tensor[D], Tensor[D]), (Output[D], Output[D])] =
    convTensorToOutput[D] * convTensorToOutput[D]

  implicit def convOutputTupToOutputTup[D: TF]: DataPipe[(Output[D], Output[D]), (Output[D], Output[D])] =
    convOutputToOutput * convOutputToOutput


  /*implicit val nil: Function.ArgType[Nil.type] = new Function.ArgType[Nil.type] {
    override def numOutputs: Int = 0
    override def outputs(arg: Nil.type): Seq[Output] = Seq.empty[Output]
    override def dataTypes(arg: Nil.type): Seq[DataType] = Seq.empty[DataType]
    override def outputsDecoder(outputs: Seq[Output]): (Nil.type, Seq[Output]) = (Nil, outputs)
    override def outputsDecoderWithKnownArg(arg: Nil.type, outputs: Seq[Output]): (Nil.type, Seq[Output]) = (Nil, outputs)
  }

  implicit def seqFuncArgTypeConstructor(size: Int): Function.ArgType[Seq[Output]] =
    new Function.ArgType[Seq[Output]] {

      override def numOutputs: Int = size//argTypes.value.map(_.value.numOutputs).sum

      override def outputs(arg: Seq[Output]): Seq[Output] = arg

      /*{
        argTypes.zip(arg).flatMap(c => c._1.value.outputs(c._2))
      }*/

      override def dataTypes(arg: Seq[Output]): Seq[DataType] = arg.map(_.dataType)

      /*{
        argTypes.zip(arg).flatMap(c => c._1.value.dataTypes(c._2))
      }*/

      override def outputsDecoder(outputs: Seq[Output]): (Seq[Output], Seq[Output]) =
        (outputs, outputs.drop(outputs.length))

      /*{

        val (decodedHead, outputsTail) = argTypes.head.value.outputsDecoder(outputs)
        val (decodedTail, tail) = argTypes.tail.map(_.value.outputsDecoder(outputsTail)).unzip
        (Seq(decodedHead) ++ decodedTail, tail.flatten)
      }*/

      override def outputsDecoderWithKnownArg(arg: Seq[Output], outputs: Seq[Output]): (Seq[Output], Seq[Output]) =
        (outputs, outputs.drop(outputs.length))

      /*{
        val (decodedHead, outputsTail) = argTypes.head.value.outputsDecoderWithKnownArg(arg.head, outputs)
        val (decodedTail, tail) = argTypes.tail.map(_.value.outputsDecoderWithKnownArg(arg.head, outputsTail)).unzip
        (Seq(decodedHead) ++ decodedTail, tail.flatten)
      }*/
    }
*/


}
