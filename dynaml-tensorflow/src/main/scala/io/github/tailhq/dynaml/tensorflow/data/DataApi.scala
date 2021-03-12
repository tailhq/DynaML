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
package io.github.tailhq.dynaml.tensorflow.data

import os.Path
import com.sksamuel.scrimage.Image
import io.github.tailhq.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.tailhq.dynaml.tensorflow.api.Api
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.UByte


private[tensorflow] object DataApi {

  val dataset: DataSet.type                      = DataSet
  val supervised_dataset: SupervisedDataSet.type = SupervisedDataSet
  val tf_dataset: TFDataSet.type                 = TFDataSet

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
  def create_image_tensor_buffered(
    buff_size: Int,
    image_to_bytes: DataPipe[Image, Array[Byte]],
    image_height: Int, image_width: Int, num_channels: Int)(
    coll: Iterable[Path], size: Int): Tensor[UByte] = {

    val load_image = StreamDataPipe(DataPipe((p: Path) => Image.fromPath(p.toNIO)) > image_to_bytes)

    println()
    val tensor_splits = coll.grouped(buff_size).toIterable.zipWithIndex.map(splitAndIndex => {

      val split_seq = splitAndIndex._1.toStream

      val progress = math.round(10*splitAndIndex._2*buff_size*100.0/size)/10d

      print("Progress %:\t")
      pprint.pprintln(progress)

      Api.tensor_from_buffer[UByte](
        split_seq.length,
        image_height, image_width, num_channels)(load_image(split_seq).flatten.toArray)

    })

    Api.concatenate(tensor_splits.toSeq, 0)
  }

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
  def create_image_tensor_buffered[Source](
    buff_size: Int, image_sources: Seq[Source],
    image_process: Map[Source, DataPipe[Image, Image]],
    images_to_bytes: DataPipe[Seq[Image], Array[Byte]],
    image_height: Int, image_width: Int, num_channels: Int)(
    coll: Iterable[Map[Source, Seq[Path]]], size: Int): Tensor[UByte] = {

    val load_image = StreamDataPipe(DataPipe((images_map: Map[Source, Seq[Path]]) => {
      image_sources.map(source => {

        val images_for_source = images_map(source).map(p => image_process(source)(Image.fromPath(p.toNIO)))

        images_to_bytes(images_for_source)
      }).toArray.flatten
    }))

    println()
    val tensor_splits = coll.grouped(buff_size).toIterable.zipWithIndex.map(splitAndIndex => {

      val split_seq = splitAndIndex._1.toStream

      val progress = math.round(10*splitAndIndex._2*buff_size*100.0/size)/10

      print("Progress %:\t")
      pprint.pprintln(progress)

      Api.tensor_from_buffer[UByte](
        split_seq.length,
        image_height, image_width, num_channels)(
        load_image(split_seq).flatten.toArray)

    })

    Api.concatenate(tensor_splits.toSeq, axis = 0)
  }

}