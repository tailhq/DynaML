package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.PDEQuadrature
import io.github.mandar2812.dynaml.tensorflow.data._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.Fetchable
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
private[dynaml] class DynamicalSystem[D <: DataType](
  val quantities: Map[String, Layer[Output, Output]],
  val dynamics: Seq[DifferentialOperator[Output, Output]],
  val input: (D, Shape),
  val target: (Seq[D], Seq[Shape]),
  val data_loss: Layer[(Output, Output), Output],
  quadrature_nodes: Tensor[D],
  quadrature_weights: Tensor[D],
  quadrature_loss_weightage: Tensor[D],
  graphInstance: Option[Graph]) {

  protected val observational_error: Layer[(Seq[Output], Seq[Output]), Output] =
    DynamicalSystem.error("ExtObsError", data_loss)

  val system_outputs: Seq[Layer[Output, Output]] = quantities.values.toSeq

  protected val quadrature: PDEQuadrature[D] =
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
  
  def solve(
    data: Seq[SupervisedDataSet[Tensor[D], Tensor[D]]],
    trainConfig: TFModel.Config,
    data_processing: TFModel.Ops = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false)(
    implicit
    evDataI: Data.Aux[Seq[Tensor[D]], Seq[Output], Seq[D], Seq[Shape]],
    evData: Data.Aux[
      (Seq[Tensor[D]], Seq[Tensor[D]]),
      (Seq[Output], Seq[Output]),
      (Seq[D], Seq[D]),
      (Seq[Shape], Seq[Shape])],
    evFunctionOutput: Function.ArgType[(Seq[Output], Seq[Output])],
    evFetchableIO: Fetchable.Aux[Seq[Output], Seq[Tensor[D]]],
    evFetchableIIO: Fetchable.Aux[(Seq[Output], Seq[Output]), (Seq[Tensor[D]], Seq[Tensor[D]])],
    ev: Estimator.SupportedInferInput[
      Seq[Tensor[D]], Seq[Tensor[D]], Seq[Tensor[D]],
      Seq[Output], Seq[D], Seq[Shape], Seq[Tensor[D]]]): DynamicalSystem.Model[D] = {

    val model = dtflearn.model[
      Seq[Tensor[D]], Seq[Output], Seq[D], Seq[Shape], Seq[Output],
      Seq[Tensor[D]], Seq[Output], Seq[D], Seq[Shape], Seq[Output],
      Seq[Tensor[D]], Seq[Tensor[D]], Seq[Tensor[D]]](
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

  protected case class ObservationalError(
    override val name: String,
    error_measure: Layer[(Output, Output), Output]) extends
    Layer[(Seq[Output], Seq[Output]), Output](name) {

    override val layerType: String = "ObservationalError"

    override def forwardWithoutContext(input: (Seq[Output], Seq[Output]))(implicit mode: Mode): Output =
      input._1.zip(input._2).map(c => error_measure.forward(c._1, c._2)).reduceLeft(_.add(_))
  }

  protected case class Model[D <: DataType](
    tfModel: TFModel[
      Seq[Tensor[D]], Seq[Output], Seq[D], Seq[Shape], Seq[Output],
      Seq[Tensor[D]], Seq[Output], Seq[D], Seq[Shape], Seq[Output],
      Seq[Tensor[D]], Seq[Tensor[D]], Seq[Tensor[D]]],
    outputs: Seq[String],
    variables: Seq[String]) {

    private val model_quantities = outputs ++ variables

    protected def predict(input: Tensor[D]): Map[String, Tensor[D]] = {

      val model_preds = tfModel.predict(Seq.fill[Tensor[D]](outputs.length)(input))

      model_quantities.zip(model_preds).toMap
    }

    def predict(quantities: String*)(input: Tensor[D]): Seq[Tensor[D]] = {
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

      override def forwardWithoutContext(input: (Seq[Output], Seq[Output]))(implicit mode: Mode): Seq[Output] =
        input._1 ++ input._2
    }

  val flatten: String => Layer[Seq[Seq[Output]], Seq[Output]] =
    (name: String) => new Layer[Seq[Seq[Output]], Seq[Output]](name) {
      override val layerType: String = "FlattenSeq"

      override def forwardWithoutContext(input: Seq[Seq[Output]])(implicit mode: Mode): Seq[Output] = input.flatten
    }

  def apply[D <: DataType](
    quantities: Map[String, Layer[Output, Output]],
    dynamics: Seq[DifferentialOperator[Output, Output]],
    input: (D, Shape),
    target: (Seq[D], Seq[Shape]),
    data_loss: Layer[(Output, Output), Output],
    quadrature_nodes: Tensor[D],
    weights: Tensor[D],
    loss_weightage: Tensor[D],
    graphInstance: Option[Graph] = None): DynamicalSystem[D] =
    new DynamicalSystem[D](
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