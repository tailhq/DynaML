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
package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.PDEQuadrature
import io.github.mandar2812.dynaml.tensorflow.data._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.Function
import org.platanios.tensorflow.api.ops.io.data.Data
import org.platanios.tensorflow.api.types.DataType

/**
  * <h3>Dynamical Systems</h3>
  * <h4>Forecasting, System Identification and Design</h4>
  *
  * @param quantities A map containing quantity names and their associated function approximations.
  * @param dynamics A system of differential operators, representing the system dynamics
  * @param input The data type and shape of the input tensor.
  * @param target The data types and shapes of the outputs and the latent variables.
  * @param data_loss The loss/error measure between the neural surrogate output and
  *                  the observations.
  * @param quadrature_nodes The co-location points in the input domain, packaged as a tensor.
  * @param quadrature_weights The weights associated with each co-location point.
  * @param quadrature_loss_weightage The weight attached to the quadrature constructed
  *                                  for approximating the integrated error between the
  *                                  surrogate and the dynamical system.
  * @param graphInstance An optional TensorFlow graph instance to create model in.
  *
  * */
class DynamicalSystem[T](
  val quantities: Map[String, Layer[Output, Output]],
  val dynamics: Seq[DifferentialOperator[Output, Output]],
  val input: (DataType.Aux[T], Shape),
  val target: (Seq[DataType.Aux[T]], Seq[Shape]),
  val data_loss: Layer[(Output, Output), Output],
  quadrature_nodes: Tensor,
  quadrature_weights: Tensor,
  quadrature_loss_weightage: Tensor,
  graphInstance: Option[Graph]) {

  protected val observational_error: Layer[(Seq[Output], Seq[Output]), Output] =
    DynamicalSystem.error("ExtObsError", data_loss)

  val system_outputs: Seq[Layer[Output, Output]] = quantities.values.toSeq

  protected val quadrature: PDEQuadrature =
    PDEQuadrature(
      "ColocationError",
      dynamics.zip(system_outputs).map(c => c._1(c._2)),
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)

  val system_variables: Seq[Map[String, DifferentialOperator[Output, Output]]] =
    dynamics.map(_.variables)

  private val data_handles = (
    tf.learn.Input(
      Seq.fill(quantities.toSeq.length)(input._1),
      Seq.fill(quantities.toSeq.length)(Shape(-1) ++ input._2),
      "Input"),
    tf.learn.Input(
      target._1,
      target._2.map(s => Shape(-1) ++ s),
      "Outputs"))

  val system_variables_mapping: Layer[Seq[Output], Seq[Output]] =
    dtflearn.seq_layer[Output, Seq[Output]](
      "SystemVariables",
      system_variables.zip(system_outputs).map(variablesAndQuantities => {

        val (variables, quantity) = variablesAndQuantities

        dtflearn.combined_layer[Output, Output]("MapVariables", variables.values.toSeq.map(_(quantity)))

      })) >>
      DynamicalSystem.flatten("FlattenVariables")

  protected val output_mapping: Layer[Seq[Output], Seq[Output]] =
    dtflearn.seq_layer[Output, Output](name ="CombinedOutputs", system_outputs)

  val model_architecture: Layer[Seq[Output], Seq[Output]] =
    dtflearn.combined_layer("CombineOutputsAndVars", Seq(output_mapping, system_variables_mapping)) >>
      DynamicalSystem.flatten("FlattenModelOutputs")

  protected val system_loss: Layer[(Seq[Output], Seq[Output]), Output] =
    observational_error >>
      quadrature >>
      tf.learn.Mean("Loss/Mean") >>
      tf.learn.ScalarSummary("Loss/Summary", "Loss")

  /**
    * Train a neural net based approximation for the
    * dynamical system.
    *
    * @param data Training data, a sequence of supervised/labeled data
    *             sets the length of the sequence must equal the number
    *             of governing equations.
    *
    * @param trainConfig Training configuration, of type [[TFModel.Config]]
    *
    * @param data_processing TensorFlow data operation pipeline, instance of [[TFModel.Ops]]
    *
    * @param inMemory Set to true if model is to be kept entirely in memory, defaults to false.
    *
    * @return A [[DynamicalSystem.Model]] which encapsulates a predictive model of type [[TFModel]]
    * */
  def solve(
    data: Seq[SupervisedDataSet[Tensor, Tensor]],
    trainConfig: TFModel.Config,
    data_processing: TFModel.Ops = TFModel.data_ops(100, 16, 10),
    inMemory: Boolean = false)(
    implicit
    evDAToDI: DataTypeAuxToDataType.Aux[Seq[DataType.Aux[T]], Seq[DataType]],
    evDToOI: DataTypeToOutput.Aux[Seq[DataType], Seq[Output]],
    evOToTI: OutputToTensor.Aux[Seq[Output], Seq[Tensor]],
    evDataI: Data.Aux[Seq[Tensor], Seq[Output], Seq[DataType], Seq[Shape]],
    evDAToD: DataTypeAuxToDataType.Aux[
      (Seq[DataType.Aux[T]], Seq[DataType.Aux[T]]),
      (Seq[DataType], Seq[DataType])],
    evData: Data.Aux[
      (Seq[Tensor], Seq[Tensor]),
      (Seq[Output], Seq[Output]),
      (Seq[DataType], Seq[DataType]),
      (Seq[Shape], Seq[Shape])],
    evOToT: OutputToTensor.Aux[(Seq[Output], Seq[Output]), (Seq[Tensor], Seq[Tensor])],
    evFunctionOutput: Function.ArgType[(Seq[Output], Seq[Output])],
    evFetchableIO: Fetchable.Aux[Seq[Output], Seq[Tensor]],
    evFetchableIIO: Fetchable.Aux[(Seq[Output], Seq[Output]), (Seq[Tensor], Seq[Tensor])],
    ev: Estimator.SupportedInferInput[
      Seq[Tensor], Seq[Tensor], Seq[Tensor],
      Seq[Output], Seq[DataType], Seq[Shape], Seq[Tensor]]): DynamicalSystem.Model[T] = {

    val model = dtflearn.model[
      Seq[Tensor], Seq[Output], Seq[DataType.Aux[T]], Seq[DataType], Seq[Shape], Seq[Output], Seq[Tensor],
      Seq[Tensor], Seq[Output], Seq[DataType.Aux[T]], Seq[DataType], Seq[Shape], Seq[Output]](
      dtfdata.supervised_dataset.collect(data).map(p => (p._1, p._2)),
      model_architecture,
      (Seq.fill(quantities.size)(input._1), Seq.fill(quantities.size)(input._2)),
      target,
      dtflearn.identity[Seq[Output]]("ID"),
      system_loss, trainConfig,
      data_processing, inMemory,
      graphInstance,
      Some(data_handles))

    model.train()

    DynamicalSystem.model(
      model,
      quantities.keys.toSeq,
      system_variables.flatMap(_.keys.toSeq)
    )
  }

}


object DynamicalSystem {

  /**
    * Observational Error loss function.
    * */
  case class ObservationalError(
    override val name: String,
    error_measure: Layer[(Output, Output), Output]) extends
    Layer[(Seq[Output], Seq[Output]), Output](name) {

    override val layerType: String = "ObservationalError"

    override protected def _forward(input: (Seq[Output], Seq[Output]))(implicit mode: Mode): Output =
      input._1.zip(input._2).map(c => error_measure.forward(c._1, c._2)).reduceLeft(_.add(_))
  }

  /**
    * Neural-net based predictive model for dynamical systems.
    *
    * @param tfModel A DynaML TensorFlow model [[TFModel]]
    *
    * @param outputs The model quantities of interest which are
    *                governed by the system dynamics.
    *
    * @param variables The unobserved quantities of the system.
    *
    * */
  case class Model[T](
    tfModel: TFModel[
      Seq[Tensor], Seq[Output], Seq[DataType.Aux[T]], Seq[DataType], Seq[Shape], Seq[Output], Seq[Tensor],
      Seq[Tensor], Seq[Output], Seq[DataType.Aux[T]], Seq[DataType], Seq[Shape], Seq[Output]],
    outputs: Seq[String],
    variables: Seq[String]) {

    private val model_quantities = outputs ++ variables

    protected def predict(input: Tensor): Map[String, Tensor] = {

      val model_preds = tfModel.predict(Seq.fill[Tensor](outputs.length)(input))

      model_quantities.zip(model_preds).toMap
    }

    def predict(quantities: String*)(input: Tensor): Seq[Tensor] = {
      require(
        quantities.forall(model_quantities.contains),
        "Each provided quantity should be in the list of model quantities,"+
          " either as an output or as an inferred variable")

      val outputs = predict(input)

      quantities.map(outputs(_))
    }


  }

  val error: ObservationalError.type = ObservationalError
  val model: Model.type              = Model

  val combine_tuple: String => Layer[(Seq[Output], Seq[Output]), Seq[Output]] =
    (name: String) => new Layer[(Seq[Output], Seq[Output]), Seq[Output]](name) {
      override val layerType: String = "CombineSeqTuple"

      override protected def _forward(input: (Seq[Output], Seq[Output]))(implicit mode: Mode): Seq[Output] =
        input._1 ++ input._2
    }

  val flatten: String => Layer[Seq[Seq[Output]], Seq[Output]] =
    (name: String) => new Layer[Seq[Seq[Output]], Seq[Output]](name) {
      override val layerType: String = "FlattenSeq"

      override protected def _forward(input: Seq[Seq[Output]])(implicit mode: Mode): Seq[Output] = input.flatten
    }

  def apply[T](
    quantities: Map[String, Layer[Output, Output]],
    dynamics: Seq[DifferentialOperator[Output, Output]],
    input: (DataType.Aux[T], Shape),
    target: (Seq[DataType.Aux[T]], Seq[Shape]),
    data_loss: Layer[(Output, Output), Output],
    quadrature_nodes: Tensor,
    weights: Tensor,
    loss_weightage: Tensor,
    graphInstance: Option[Graph] = None): DynamicalSystem[T] =
    new DynamicalSystem[T](
      quantities,
      dynamics,
      input,
      target,
      data_loss,
      quadrature_nodes,
      weights,
      loss_weightage,
      graphInstance
    )
  
}