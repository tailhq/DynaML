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

import io.github.mandar2812.dynaml.pipes.DataPipe
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, OutputToTensor}
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}

class DataSet[X](val data: Iterable[X]) {
  self =>

  lazy val size: Int = data.toSeq.length

  def map[Y](pipe: DataPipe[X, Y]): DataSet[Y] = new DataSet[Y](data.map(pipe(_)))

  def flatMap[Y](pipe: DataPipe[X, Iterable[Y]]): DataSet[Y] = new DataSet[Y](data.flatMap(pipe(_)))

  def zip[Y](other: DataSet[Y]): ZipDataSet[X, Y] = new ZipDataSet[X, Y](self, other)

  def concatenate(other: DataSet[X]): DataSet[X] = new DataSet[X](self.data ++ other.data)

  def partition(f: DataPipe[X, Boolean]): TFDataSet[X] = {
    val data_split = data.partition(f(_))

    TFDataSet(new DataSet(data_split._1), new DataSet(data_split._2))
  }

  def build[T, O, DA, D, S](
    transform: Either[DataPipe[X, T], DataPipe[X, O]],
    dataType: DA, shape: S = null)(
    implicit
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, O, D, S],
    evOToT: OutputToTensor.Aux[O, T],
    evFunctionOutput: Function.ArgType[O]
  ): Dataset[T, O, D, S] = transform match {
    case Left(pipe) => tf.data.fromGenerator(
      () => self.data.map(pipe(_)),
      dataType, shape)
    case Right(pipe) => self.data
      .map(x => tf.data.OutputDataset(pipe(x)))
      .reduceLeft[Dataset[T, O, D, S]](
      (a, b) => a.concatenate(b)
    )
  }

}

class ZipDataSet[X, Y](
  val dataset1: DataSet[X],
  val dataset2: DataSet[Y]) extends
  DataSet[(X, Y)](dataset1.data.zip(dataset2.data)) {

  def unzip: (DataSet[X], DataSet[Y]) = (dataset1, dataset2)

}

/**
  * <h3>Supervised Data Set</h3>
  *
  *
  * */
case class SupervisedDataSet[X, Y](
  features: DataSet[X],
  targets: DataSet[Y]) extends
  ZipDataSet[X, Y](features, targets)



/**
  *
  *
  * */
case class TFDataSet[T](
  training_dataset: DataSet[T],
  test_dataset: DataSet[T])