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
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.ops.data._

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


  private def filter(filterFn: X => Boolean): DataSet[X] = DataSet[X](data.filter(filterFn))

  /**
    * Filter elements of this data set which satisfy
    * a predicate.
    * */
  def filter(pipe: DataPipe[X, Boolean]): DataSet[X] = filter(pipe.run _)


  private def filterNot(filterFn: X => Boolean): DataSet[X] = DataSet[X](data.filterNot(filterFn))

  /**
    * Filter elements of this data set which does not
    * satisfy a predicate.
    * */
  def filterNot(pipe: DataPipe[X, Boolean]): DataSet[X] = filterNot(pipe.run _)

  /**
    * Creates a new data set of type [[Y]]
    * */
  private def map[Y](func: X => Y): DataSet[Y] = DataSet[Y](data.map(func))

  /**
    * Creates a new data set of type [[Y]]
    * */
  def map[Y](pipe: DataPipe[X, Y]): DataSet[Y] = map(pipe.run _)

  /**
    * Maps each element into a collection of elements of type [[Y]],
    * and then concatenates each resulting collection into a single
    * data set.
    * */
  private def flatMap[Y](func: X => Iterable[Y]): DataSet[Y] = DataSet[Y](data.flatMap(func))

  /**
    * Maps each element into a collection of elements of type [[Y]],
    * and then concatenates each resulting collection into a single
    * data set.
    * */
  def flatMap[Y](pipe: DataPipe[X, Iterable[Y]]): DataSet[Y] = flatMap(pipe.run _)

  /**
    * Create a data set consisting of ([[X]], [[Y]]) pairs.
    * */
  def zip[Y](other: DataSet[Y]): ZipDataSet[X, Y] = ZipDataSet[X, Y](self, other)

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

  def scanLeft[Y](z: Y)(scanPipe: DataPipe2[Y, X, Y]): DataSet[Y] = DataSet(data.scanLeft(z)(scanPipe(_, _)))

  def scanRight[Y](z: Y)(scanPipe: DataPipe2[X, Y, Y]): DataSet[Y] = DataSet(data.scanRight(z)(scanPipe(_, _)))

  def scan[Y >: X](z: Y)(scanPipe: DataPipe2[Y, Y, Y]): DataSet[Y] = DataSet(data.scan(z)(scanPipe(_, _)))

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

  def to_zip[Y, Z](f: DataPipe[X, (Y, Z)]): ZipDataSet[Y, Z] = {
    val data_split = data.map(f(_)).unzip
    ZipDataSet[Y, Z](DataSet[Y](data_split._1), DataSet[Z](data_split._2))
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
    * @tparam D The type of the data type objects for each data element.
    * @tparam S The type of the object representing the shape of the data tensors.
    *
    * @param transformation A data pipe from [[X]] to [[T]]
    * @param dataType The data type of the underlying patterns.
    * @param shape The shape of the data patterns, defaults to null, i.e. is
    *              inferred during run time.
    *
    * @return A TensorFlow data set handle.
    * */
  def build[T, O, D, S](
    transformation: DataPipe[X, T],
    dataType: D, shape: S = null)(
    implicit
    evTensorToOutput: TensorToOutput.Aux[T, O],
    evOutputToDataType: OutputToDataType.Aux[O, D],
    evDataTypeToShape: DataTypeToShape.Aux[D, S],
    evOutputToShape: OutputToShape.Aux[O, S],
    evOutputStructure: OutputStructure[O]): Dataset[O] =
    tf.data.datasetFromGenerator[O, T, D, S](() => self.map(transformation).data, dataType, shape)

  protected def build[T, O, D, S](
    transformation: DataPipe[Iterable[X], Iterable[O]],
    dataType: D, shape: S)(
    implicit
    evOutputStructure: OutputStructure[O],
    evOutputToDataType: OutputToDataType.Aux[O, D],
    evOutputToShape: OutputToShape.Aux[O, S]
  ): Dataset[O] =
    self
      .transform(transformation)
      .map(DataPipe((batch: O) => tf.data.datasetFromOutputSlices(batch)))
      .reduceLeft[Dataset[O]](
      DataPipe2((l: Dataset[O], r: Dataset[O]) => l.concatenateWith(r)))


  def build_buffered[T, O, D, S](
    buffer_size: Int,
    stackOp: DataPipe[Iterable[O], O],
    dataType: D,
    shape: S = null)(
    implicit
    convertToOutput: DataPipe[X, O],
    evOutputStructure: OutputStructure[O],
    evOutputToDataType: OutputToDataType.Aux[O, D],
    evOutputToShape: OutputToShape.Aux[O, S]): Dataset[O] = {

    val buffer_and_stack =
      DataPipe((d: Iterable[X]) => d.grouped(buffer_size).toIterable) >
        IterableDataPipe(IterableDataPipe(convertToOutput)) >
        IterableDataPipe(stackOp)


    build(buffer_and_stack, dataType, shape)
  }

}

object DataSet {

  def apply[X](data: Iterable[X]): DataSet[X] = new DataSet(data)

  /**
    * Collect a sequence of data sets into a single data set.
    *
    * @tparam X The type of each data instance.
    * @param datasets A sequence of [[DataSet]] objects.
    *
    * @return A [[DataSet]] over sequence of [[X]]
    * */
  def collect[X](datasets: Seq[DataSet[X]]): DataSet[Seq[X]] = {
    require(datasets.map(_.size) == Seq.fill(datasets.length)(datasets.head.size))

    apply(Iterable.tabulate(datasets.head.size)(i => datasets.map(d => d.data.toSeq(i))))
  }

}

case class OutputDataSet[T: TF](override val data: Iterable[Output[T]]) extends
  DataSet[Output[T]](data) {

  self =>

  def build(): Dataset[Output[T]] = tf.data.datasetFromOutputs(tf.concatenate(self.data.toSeq))


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

  /**
    * Join the current data set to another key value data set.
    * Join operation is carried out over keys of type [[X]].
    *
    * */
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

  /**
    * Split into training and test sets.
    * */
  override def partition(f: DataPipe[(X, Y), Boolean]): TFDataSet[(X, Y)] = {
    val data_split = data.partition(f(_))

    val (features_train, targets_train) = data_split._1.unzip

    val (features_test, targets_test)   = data_split._2.unzip

    TFDataSet(
      SupervisedDataSet(new DataSet[X](features_train), new DataSet[Y](targets_train)),
      SupervisedDataSet(new DataSet[X](features_test),  new DataSet[Y](targets_test)))
  }


}

object SupervisedDataSet {

  def apply[X, Y](features: Iterable[X], targets: Iterable[Y]): SupervisedDataSet[X, Y] =
    SupervisedDataSet(DataSet(features), DataSet(targets))

  def apply[X, Y](data: Iterable[(X, Y)]): SupervisedDataSet[X, Y] = {

    val (features, targets) = data.unzip

    SupervisedDataSet(DataSet(features), DataSet(targets))
  }

  /**
    * Collect a sequence of supervised data sets into a single data set.
    *
    * @tparam X The type of the data features/inputs.
    * @tparam Y The type of the data outputs/targets.
    * @param datasets A sequence of [[SupervisedDataSet]] objects.
    *
    * @return A [[SupervisedDataSet]] over sequences of [[X]] and  [[Y]] respectively.
    * */
  def collect[X, Y](datasets: Seq[SupervisedDataSet[X, Y]]): SupervisedDataSet[Seq[X], Seq[Y]] = {
    require(datasets.map(_.size) == Seq.fill(datasets.length)(datasets.head.size))

    SupervisedDataSet(
      DataSet.collect[X](datasets.map(_.features)),
      DataSet.collect[Y](datasets.map(_.targets)))
  }

}


case class TFDataSet[T](
  training_dataset: DataSet[T],
  test_dataset: DataSet[T])