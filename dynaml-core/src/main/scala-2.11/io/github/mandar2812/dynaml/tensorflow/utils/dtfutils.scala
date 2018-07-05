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

import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.pipes._

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>DynaML Tensorflow Utilities Package</h3>
  *
  * Contains miscellaneous utility functions for DynaML's
  * tensorflow handle/package.
  *
  * */
object dtfutils {

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq(t: Tensor): Iterator[Double] = {
    val datatype = t.dataType.toString()
    t.entriesIterator.map(x =>
      if(datatype == "FLOAT64") x.asInstanceOf[Double]
      else x.asInstanceOf[Float].toDouble)
  }


  def get_ffstack_properties(
    neuron_counts: Seq[Int],
    ff_index: Int): (Seq[Shape], Seq[String], Seq[String]) = {

    val layer_parameter_names = (ff_index until ff_index + neuron_counts.length - 1).map(i => "Linear_"+i+"/Weights")
    val layer_shapes          = neuron_counts.sliding(2).toSeq.map(c => Shape(c.head, c.last))
    val layer_datatypes       = Seq.fill(layer_shapes.length)("FLOAT64")


    (layer_shapes, layer_parameter_names, layer_datatypes)
  }

  /**
    * Calculate the Kullback Leibler divergence of
    * a probability density from a prior density.
    * */
  def kl(prior: Output, p: Output): Output =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

  def kl(prior: Tensor, p: Tensor): Output =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

  /**
    * Calculate the Jensen Shannon divergence
    * between a probability and a target probability.
    * */
  def js(target_prob: Output, prob: Output) = {
    val m = target_prob.add(prob).divide(2.0)
    kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
  }

  def js(target_prob: Tensor, prob: Tensor) = {
    val m = target_prob.add(prob).divide(2.0)
    kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
  }

  /**
    * Calculate the Hellinger distance between two
    * probability distributions.
    * */
  def hellinger(target_prob: Output, prob: Output) =
    target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))

  def hellinger(target_prob: Tensor, prob: Tensor) =
    target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))

  def cross_entropy(target_prob: Output, prob: Output) =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()

  /**
    * Calculate the cross-entropy of two
    * probability distributions.
    * */
  def cross_entropy(target_prob: Tensor, prob: Tensor) =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()

  /**
    * Create a tensor from a collection of image data,
    * in a buffered manner.
    *
    * @param buff_size The size of the buffer (in number of images to load at once)
    * @param image_height The height, in pixels, of the image.
    * @param image_width The width, in pixels, of the image.
    * @param num_channels The number of channels in the image data.
    * @param coll The collection which holds the data for each image.
    * @param size The number of elements in the collection
    * */
  def create_image_tensor_buffered(buff_size: Int)(
    image_height: Int, image_width: Int, num_channels: Int)(
    coll: Iterable[Array[Byte]], size: Int): Tensor = {

    println()
    val tensor_splits = coll.grouped(buff_size).toIterable.zipWithIndex.map(splitAndIndex => {

      val split_seq = splitAndIndex._1.toSeq

      val progress = splitAndIndex._2*buff_size*100.0/size

      print("Progress %:\t")
      pprint.pprintln(progress)

      dtf.tensor_from(
        dtype = "UINT8", split_seq.length,
        image_height, image_width, num_channels)(
        split_seq.toArray.flatten[Byte])

    })

    dtf.concatenate(tensor_splits.toSeq, axis = 0)
  }


  def buffered_preds_helper[
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, EI,
  InferInput, InferOutput,
  ModelInferenceOutput](
    predictiveModel: Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI],
    workingData: InferInput,
    buffer: Int, dataSize: Int)(
    implicit getSplitByIndex: MetaPipe12[InferInput, Int, Int, InferInput],
    concatenateSplits: DataPipe[Iterable[InferOutput], InferOutput],
    evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): Some[InferOutput] = {

    val get_data_split = getSplitByIndex(workingData)

    val preds_splits: Iterable[InferOutput] = (0 until dataSize)
      .grouped(buffer)
      .map(indices =>
        predictiveModel.infer[InferInput, InferOutput, ModelInferenceOutput](
          () => get_data_split(indices.head, indices.last)))
      .toIterable

    Some(concatenateSplits(preds_splits))
  }

 def buffered_preds[
 IT, IO, ID, IS, I,
 TT, TO, TD, TS, EI](
   predictiveModel: Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI])(
   data: AbstractDataSet[IT, IO, ID, IS, TT, TO, TD, TS],
   pred_flags: (Boolean, Boolean) = (false, true),
   buff_size: Int = 400)(
   implicit getSplitByIndex: MetaPipe12[IT, Int, Int, IT],
   concatenateSplits: DataPipe[Iterable[I], I],
   evFetchableIO: Fetchable.Aux[IO, IT],
   evFetchableI: Fetchable.Aux[I, IT],
   evFetchableIIO: Fetchable.Aux[(IO, I), (IT, IT)],
   ev: Estimator.SupportedInferInput[IT, I, IT, IO, ID, IS, IT]
 ): (Option[I], Option[I]) = {

    val train_preds =
      if (pred_flags._1) buffered_preds_helper(predictiveModel, data.trainData, buff_size, data.nTrain)
      else None

    val test_preds =
      if (pred_flags._2) buffered_preds_helper(predictiveModel, data.testData, buff_size, data.nTest)
      else None

    (train_preds, test_preds)
  }

}

