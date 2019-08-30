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
package io.github.mandar2812.dynaml.tensorflow.evaluation

import io.github.mandar2812.dynaml.pipes._
import org.platanios.tensorflow.api._

import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}

/**
  * Top level class for metrics computed on (eager) Tensorflow objects.
  *
  * @param preds Predictions
  *
  * @param targets The actual output values.
  * */
abstract class MetricsTF[D: TF](
  val names: Seq[String],
  val preds: Tensor[D],
  val targets: Tensor[D]) {

  implicit val formats = DefaultFormats

  protected val scoresAndLabels: (Tensor[D], Tensor[D]) = (preds, targets)

  protected var name = "Target"

  lazy val results: Tensor[D] = run()

  def _target_quantity: String = name

  def target_quantity_(n: String): Unit = {
    name = n
  }

  def print(): Unit = {
    println("\nModel Performance: " + name)
    println("============================")
    println()

    names.zipWithIndex.foreach(n => {

      val value: Tensor[D] = results(n._2, ---)

      val metric = n._1

      println(
        metric + ": " + value
          .summarize(maxEntries = value.size.toInt, flattened = true)
      )
      println()
    })
  }

  def generatePlots(): Unit = {}

  /**
    * Has the actual computational logic of producing
    * the metrics which are to be calculated.
    *
    * Implement this method in sub-classes.
    * */
  protected def run(): Tensor[D]

  def to_json: String = {

    val metrics = tfi.unstack(run(), number = names.length, axis = 0)

    val results: Map[String, Any] =
      names.zip(metrics.map(_.entriesIterator.toIndexedSeq)).toMap ++
        Map("shape"    -> metrics.head.shape.entriesIterator.toIndexedSeq) ++
        Map("quantity" -> name)

    write_json(results)
  }

}

object MetricsTF {

  def apply[EvalIn](
    compute_batch: DataPipe3[EvalIn, Option[Output[Float]], String, Output[
      Float
    ]],
    compute_streaming: DataPipe3[EvalIn, Option[Output[Float]], String, tf.metrics.Metric.StreamingInstance[
      Output[Float]
    ]],
    id: String = "performance"
  ): tf.metrics.Metric[EvalIn, Output[Float]] =
    new ops.metrics.Metric[EvalIn, Output[Float]] {

      override def name: String = id

      override def compute(
        values: EvalIn,
        weights: Option[Output[Float]],
        name: String = s"$name/Compute"
      ): Output[Float] = compute_batch(values, weights, name)

      override def streaming(
        values: EvalIn,
        weights: Option[Output[Float]],
        name: String = s"$name/Streaming"
      ): ops.metrics.Metric.StreamingInstance[Output[Float]] =
        compute_streaming(values, weights, name)

    }

}

/**
  * Computes any performance score which is averaged
  * over a data set.
  *
  * @param nameScope The string identifier to use for this score,
  *                  the summaries if saved will be saved under this tag.
  *
  * @param compute A [[DataPipe]] which computes the score for a single minibatch.
  * */
class Performance[EvalIn](
  val nameScope: String,
  val compute: DataPipe[EvalIn, Output[Float]],
  protected val defaultWeights: Option[Tensor[Float]] = None,
  val variablesCollections: Set[Graph.Key[Variable[Any]]] =
    Set(tf.metrics.Metric.METRIC_VARIABLES),
  val valuesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_VALUES),
  val updatesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_UPDATES),
  val resetsCollections: Set[Graph.Key[UntypedOp]] =
    Set(tf.metrics.Metric.METRIC_RESETS))
    extends tf.metrics.Metric[EvalIn, Output[Float]] {

  override def name: String = nameScope

  private[this] val meanMetric = {
    tf.metrics.Mean(
      name,
      defaultWeights,
      variablesCollections,
      valuesCollections,
      updatesCollections,
      resetsCollections
    )
  }

  override def compute(
    values: EvalIn,
    weights: Option[Output[Float]],
    name: String = s"$name/Compute"
  ): Output[Float] =
    meanMetric.compute(compute(values), weights, name)

  override def streaming(
    values: EvalIn,
    weights: Option[Output[Float]],
    name: String = s"$name/Streaming"
  ): tf.metrics.Metric.StreamingInstance[Output[Float]] =
    meanMetric.streaming(compute(values), weights, name)

}

object Performance {

  def apply[EvalIn](
    nameScope: String,
    compute: DataPipe[EvalIn, Output[Float]],
    defaultWeights: Option[Tensor[Float]] = None,
    variablesCollections: Set[Graph.Key[Variable[Any]]] =
      Set(tf.metrics.Metric.METRIC_VARIABLES),
    valuesCollections: Set[Graph.Key[Output[Any]]] =
      Set(tf.metrics.Metric.METRIC_VALUES),
    updatesCollections: Set[Graph.Key[Output[Any]]] =
      Set(tf.metrics.Metric.METRIC_UPDATES),
    resetsCollections: Set[Graph.Key[UntypedOp]] =
      Set(tf.metrics.Metric.METRIC_RESETS)
  ): Performance[EvalIn] =
    new Performance[EvalIn](
      nameScope,
      compute,
      defaultWeights,
      variablesCollections,
      valuesCollections,
      updatesCollections,
      resetsCollections
    )
}

/**
  * Computes the mean square error (MSE score).
  * */
case class MSE[I, T: TF: IsFloatOrDouble](
  override val defaultWeights: Option[Tensor[Float]] = None,
  override val variablesCollections: Set[Graph.Key[Variable[Any]]] =
    Set(tf.metrics.Metric.METRIC_VARIABLES),
  override val valuesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_VALUES),
  override val updatesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_UPDATES),
  override val resetsCollections: Set[Graph.Key[UntypedOp]] =
    Set(tf.metrics.Metric.METRIC_RESETS))
    extends Performance[(Output[T], (I, Output[T]))](
      "MSE",
      DataPipe[(Output[T], (I, Output[T])), Output[Float]](
        c => c._1.subtract(c._2._2).square.mean(axes = 1).castTo[Float]
      ),
      defaultWeights,
      variablesCollections,
      valuesCollections,
      updatesCollections,
      resetsCollections
    )

/**
  * Computes the mean absolute error (MAE score).
  * */    
  case class MAE[I, T: TF: IsFloatOrDouble](
  override val defaultWeights: Option[Tensor[Float]] = None,
  override val variablesCollections: Set[Graph.Key[Variable[Any]]] =
    Set(tf.metrics.Metric.METRIC_VARIABLES),
  override val valuesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_VALUES),
  override val updatesCollections: Set[Graph.Key[Output[Any]]] =
    Set(tf.metrics.Metric.METRIC_UPDATES),
  override val resetsCollections: Set[Graph.Key[UntypedOp]] =
    Set(tf.metrics.Metric.METRIC_RESETS))
    extends Performance[(Output[T], (I, Output[T]))](
      "MSE",
      DataPipe[(Output[T], (I, Output[T])), Output[Float]](
        c => c._1.subtract(c._2._2).abs.mean(axes = 1).castTo[Float]
      ),
      defaultWeights,
      variablesCollections,
      valuesCollections,
      updatesCollections,
      resetsCollections
    )
