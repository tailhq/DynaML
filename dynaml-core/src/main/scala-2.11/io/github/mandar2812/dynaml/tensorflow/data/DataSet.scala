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

/**
  * <h3>DynaML Data Set</h3>
  *
  * The data set class, wraps an underlying
  * collection of elements of type [[X]].
  *
  * It can be used to create an object which
  * can access a potentially large number
  * of data patterns.
  *
  * It is also possible to transform the
  * data collection in the classical functional
  * paradigm of [[map()]], [[flatMap()]].
  *
  * @tparam X The type of each data pattern.
  *
  * @param data The underlying data collection,
  *             represented as an [[Iterable]]
  *             of elements, each of type [[X]].
  *
  * @author mandar2812 date 2018/07/17
  * */
class DataSet[X](val data: Iterable[X]) {
  self =>

  lazy val size: Int = data.toSeq.length

  /**
    * Filter elements of this data set which satisfy
    * a predicate.
    * */
  def filter(pipe: DataPipe[X, Boolean]): DataSet[X] = new DataSet[X](self.data.filter(pipe(_)))

  /**
    * Filter elements of this data set which does not
    * satisfy a predicate.
    * */
  def filterNot(pipe: DataPipe[X, Boolean]): DataSet[X] = new DataSet[X](self.data.filterNot(pipe(_)))

  /**
    * Creates a new data set of type [[Y]]
    * */
  def map[Y](pipe: DataPipe[X, Y]): DataSet[Y] = new DataSet[Y](data.map(pipe(_)))

  /**
    * Maps each element into a collection of elements of type [[Y]],
    * and then concatenates each resulting collection into a single
    * data set.
    * */
  def flatMap[Y](pipe: DataPipe[X, Iterable[Y]]): DataSet[Y] = new DataSet[Y](data.flatMap(pipe(_)))

  /**
    * Create a data set consisting of ([[X]], [[Y]]) pairs.
    * */
  def zip[Y](other: DataSet[Y]): ZipDataSet[X, Y] = new ZipDataSet[X, Y](self, other)

  /**
    * Join the current data collection with another collection
    * */
  def concatenate(other: DataSet[X]): DataSet[X] = new DataSet[X](self.data ++ other.data)

  /**
    * Split the data collection into a train-test split.
    *
    * @return A result of type [[TFDataSet]], containing
    *         both the training and test splits.
    * */
  def partition(f: DataPipe[X, Boolean]): TFDataSet[X] = {
    val data_split = data.partition(f(_))

    TFDataSet(new DataSet(data_split._1), new DataSet(data_split._2))
  }

  /**
    * Convert the current collection into an instance
    * of [[SupervisedDataSet]].
    * */
  def to_supervised[Y, Z](f: DataPipe[X, (Y, Z)]): SupervisedDataSet[Y, Z] = {
    val data_split = data.map(f(_)).unzip
    SupervisedDataSet[Y, Z](new DataSet[Y](data_split._1), new DataSet[Z](data_split._2))
  }

  /**
    * Construct a TensorFlow data set, from
    * the current data collection
    *
    * @tparam T The tensor type.
    * @tparam O Symbolic tensor (output) type.
    * @tparam DA The type of the auxiliary data structure
    * @tparam D The type of the data type objects for each data element.
    * @tparam S The type of the object representing the shape of the data tensors.
    *
    * @param transform Either a data pipe from [[X]] to [[T]] or from [[X]] to [[O]]
    * @param dataType The data type of the underlying patterns.
    * @param shape The shape of the data patterns, defaults to null, i.e. is
    *              inferred during run time.
    *
    * @return A TensorFlow data set handle.
    * */
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

object DataSet {
  def apply[X](data: Iterable[X]): DataSet[X] = new DataSet(data)
}

/**
  * A data collection consisting of ([[X]], [[Y]]) pairs.
  * */
class ZipDataSet[X, Y](
  val dataset1: DataSet[X],
  val dataset2: DataSet[Y]) extends
  DataSet[(X, Y)](dataset1.data.zip(dataset2.data)) {

  def unzip: (DataSet[X], DataSet[Y]) = (dataset1, dataset2)

}

/**
  * <h3>Supervised Data Set</h3>
  *
  * A data collection with features of type [[X]] and
  * targets of type [[Y]], suitable for supervised learning
  * tasks.
  *
  * */
case class SupervisedDataSet[X, Y](
  features: DataSet[X],
  targets: DataSet[Y]) extends
  ZipDataSet[X, Y](features, targets) {

  self  =>

  override def partition(f: DataPipe[(X, Y), Boolean]): TFDataSet[(X, Y)] = {
    val data_split = data.partition(f(_))

    val (features_train, targets_train) = data_split._1.unzip

    val (features_test, targets_test)   = data_split._2.unzip

    TFDataSet(
      SupervisedDataSet(new DataSet[X](features_train), new DataSet[Y](targets_train)),
      SupervisedDataSet(new DataSet[X](features_test),  new DataSet[Y](targets_test)))
  }


}


case class TFDataSet[T](
  training_dataset: DataSet[T],
  test_dataset: DataSet[T])