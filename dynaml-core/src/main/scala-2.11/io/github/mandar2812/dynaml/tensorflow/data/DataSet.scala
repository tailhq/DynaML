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

import io.github.mandar2812.dynaml.pipes._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, OutputToTensor}
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset, OutputDataset, OutputSlicesDataset}

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
  def filter(pipe: DataPipe[X, Boolean]): DataSet[X] = DataSet[X](self.data.filter(pipe(_)))

  /**
    * Filter elements of this data set which does not
    * satisfy a predicate.
    * */
  def filterNot(pipe: DataPipe[X, Boolean]): DataSet[X] = DataSet[X](self.data.filterNot(pipe(_)))

  /**
    * Creates a new data set of type [[Y]]
    * */
  def map[Y](pipe: DataPipe[X, Y]): DataSet[Y] = DataSet[Y](data.map(pipe(_)))

  def map(pipe: DataPipe[X, Output]): OutputDataSet = OutputDataSet(data.map(pipe(_)))

  /**
    * Maps each element into a collection of elements of type [[Y]],
    * and then concatenates each resulting collection into a single
    * data set.
    * */
  def flatMap[Y](pipe: DataPipe[X, Iterable[Y]]): DataSet[Y] = DataSet[Y](data.flatMap(pipe(_)))

  /**
    * Create a data set consisting of ([[X]], [[Y]]) pairs.
    * */
  def zip[Y](other: DataSet[Y]): ZipDataSet[X, Y] = new ZipDataSet[X, Y](self, other)

  /**
    * Join the current data collection with another collection
    * */
  def concatenate(other: DataSet[X]): DataSet[X] = DataSet[X](self.data ++ other.data)

  /**
    * Transform the underlying collection in a way that uses potentially all of its elements.
    * */
  def transform[Y](transformation: DataPipe[Iterable[X], Iterable[Y]]): DataSet[Y] = DataSet[Y](transformation(data))

  def grouped(num: Int): DataSet[Seq[X]] = transform(
    DataPipe((d: Iterable[X]) => d.grouped(num).toIterable.map(_.toSeq))
  )

  def reduce[Y](transformation: DataPipe[Iterable[X], Y]): Y = transformation(data)

  def reduce[Y >: X](reducePipe: DataPipe2[Y, Y, Y]): Y = data.reduce[Y](reducePipe(_, _))

  def reduceLeft[Y >: X](reducePipe: DataPipe2[Y, X, Y]): Y = data.reduceLeft[Y](reducePipe(_, _))

  /**
    * Split the data collection into a train-test split.
    *
    * @return A result of type [[TFDataSet]], containing
    *         both the training and test splits.
    * */
  def partition(f: DataPipe[X, Boolean]): TFDataSet[X] = {
    val data_split = data.partition(f(_))

    TFDataSet(DataSet(data_split._1), DataSet(data_split._2))
  }

  /**
    * Convert the current collection into an instance
    * of [[SupervisedDataSet]].
    * */
  def to_supervised[Y, Z](f: DataPipe[X, (Y, Z)]): SupervisedDataSet[Y, Z] = {
    val data_split = data.map(f(_)).unzip
    SupervisedDataSet[Y, Z](DataSet[Y](data_split._1), DataSet[Z](data_split._2))
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
    * @param transformation Either a data pipe from [[X]] to [[T]] or from [[X]] to [[O]]
    * @param dataType The data type of the underlying patterns.
    * @param shape The shape of the data patterns, defaults to null, i.e. is
    *              inferred during run time.
    *
    * @return A TensorFlow data set handle.
    * */
  def build[T, O, DA, D, S](
    transformation: Either[DataPipe[X, T], DataPipe[X, O]],
    dataType: DA, shape: S)(
    implicit
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, O, D, S],
    evOToT: OutputToTensor.Aux[O, T],
    evFunctionOutput: Function.ArgType[O]
  ): Dataset[T, O, D, S] = transformation match {
    case Left(pipe) => tf.data.fromGenerator(
      () => self.data.map(pipe(_)),
      dataType, shape)
    case Right(pipe) => self.data
      .map(x => tf.data.OutputDataset(pipe(x)))
      .reduceLeft[Dataset[T, O, D, S]](
      (a, b) => a.concatenate(b)
    )
  }

  def build[T, O, DA, D, S](
    transformation: DataPipe[Iterable[X], Iterable[Iterable[O]]],
    dataType: DA, shape: S)(
    implicit
    concatOp: DataPipe[Iterable[O], O],
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, O, D, S],
    evOToT: OutputToTensor.Aux[O, T],
    evFunctionOutput: Function.ArgType[O]): Dataset[T, O, D, S] =
    self
      .transform(transformation)
      .map(concatOp)
      .map(DataPipe((batch: O) => tf.data.OutputSlicesDataset[T, O, D, S](batch)))
      .reduceLeft(DataPipe2((l: Dataset[T, O, D, S], r: OutputSlicesDataset[T, O, D, S]) => l.concatenate(r)))


  def build[T, O, DA, D, S](
    buffer_size: Int,
    dataType: DA,
    shape: S = null)(
    implicit
    convertToOutput: DataPipe[X, O],
    concatOp: DataPipe[Iterable[O], O],
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, O, D, S],
    evOToT: OutputToTensor.Aux[O, T],
    evFunctionOutput: Function.ArgType[O]): Dataset[T, O, D, S] = {

    val buffer_and_concat =
      DataPipe((d: Iterable[X]) => d.grouped(buffer_size).toIterable) >
        IterableDataPipe(IterableDataPipe(convertToOutput))

    build(buffer_and_concat, dataType, shape)
  }

}

object DataSet {
  def apply[X](data: Iterable[X]): DataSet[X] = new DataSet(data)
}

case class OutputDataSet(override val data: Iterable[Output]) extends
  DataSet[Output](data) {

  self =>

  def build[T, DA, D, S](
    transform: DataPipe[Output, Output],
    dataType: DA, shape: S)(
    implicit
    evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    evData: Data.Aux[T, Output, D, S],
    evOToT: OutputToTensor.Aux[Output, T],
    evFunctionOutput: Function.ArgType[Output]): Dataset[T, Output, D, S] =
    tf.data.OutputSlicesDataset(tf.concatenate(self.data.toSeq))


}

/**
  * A data collection consisting of ([[X]], [[Y]]) pairs.
  * */
class ZipDataSet[X, Y](
  val dataset1: DataSet[X],
  val dataset2: DataSet[Y]) extends
  DataSet[(X, Y)](dataset1.data.zip(dataset2.data)) {

  self =>

  def unzip: (DataSet[X], DataSet[Y]) = (dataset1, dataset2)

  def join[Z](other: ZipDataSet[X, Z]): ZipDataSet[X, (Y, Z)] = {

    val otherMap = other.data.toMap

    val joined_data = self.data.map(pattern => {
      (pattern._1, (pattern._2, otherMap.get(pattern._1)))
    }).filter(_._2._2.isDefined)
      .map(p => (p._1, (p._2._1, p._2._2.get))).unzip

    ZipDataSet(joined_data._1, joined_data._2)
  }

}

object ZipDataSet {

  def apply[X, Y](
    dataset1: DataSet[X],
    dataset2: DataSet[Y]): ZipDataSet[X, Y] =
    new ZipDataSet(dataset1, dataset2)

  def apply[X, Y](
    dataset1: Iterable[X],
    dataset2: Iterable[Y]): ZipDataSet[X, Y] =
    new ZipDataSet(
      DataSet(dataset1),
      DataSet(dataset2)
    )
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