package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.PDEQuadrature
import io.github.mandar2812.dynaml.tensorflow.data._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsFloat32OrFloat64, IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.Function

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
private[dynaml] class DynamicalSystem[D : TF: IsNotQuantized: IsFloat32OrFloat64](
  val quantities: Map[String, Layer[Output[D], Output[D]]],
  val dynamics: Seq[DifferentialOperator[Output[D], Output[D]]],
  val input: (D, Shape),
  val target: (Seq[D], Seq[Shape]),
  val data_loss: Layer[(Output[D], Output[D]), Output[D]],
  quadrature_nodes: Tensor[D],
  quadrature_weights: Tensor[D],
  quadrature_loss_weightage: Tensor[D],
  graphInstance: Option[Graph])(
  implicit
  evStructureI: NestedStructure.Aux[Seq[Output[D]], Seq[Tensor[D]], Seq[D], Seq[Shape]],
  evStructure: NestedStructure.Aux[
    (Seq[Output[D]], Seq[Output[D]]),
    (Seq[Tensor[D]], Seq[Tensor[D]]),
    (Seq[D], Seq[D]), (Seq[Shape], Seq[Shape])]) {

  protected val observational_error: Layer[(Seq[Output[D]], Seq[Output[D]]), Output[D]] =
    DynamicalSystem.error("ExtObsError", data_loss)

  val system_outputs: Seq[Layer[Output[D], Output[D]]] = quantities.values.toSeq

  protected val quadrature: PDEQuadrature[D] =
    PDEQuadrature(
      "ColocationError",
      dynamics.zip(system_outputs).map(c => c._1(c._2)),
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)

  val system_variables: Seq[Map[String, DifferentialOperator[Output[D], Output[D]]]] =
    dynamics.map(_.variables)

  private val data_handles = (
    tf.learn.Input[Seq[Output[D]], Seq[Tensor[D]], Seq[D], Seq[Shape]](
      Seq.fill(quantities.toSeq.length)(input._1),
      Seq.fill(quantities.toSeq.length)(Shape(-1) ++ input._2),
      "Input"),
    tf.learn.Input[Seq[Output[D]], Seq[Tensor[D]], Seq[D], Seq[Shape]](
      target._1,
      target._2.map(s => Shape(-1) ++ s),
      "Outputs"))

  val system_variables_mapping: Layer[Seq[Output[D]], Seq[Output[D]]] =
    dtflearn.seq_layer[Output[D], Seq[Output[D]]](
      "SystemVariables",
      system_variables.zip(system_outputs).map(variablesAndQuantities => {

        val (variables, quantity) = variablesAndQuantities

        dtflearn.combined_layer[Output[D], Output[D]]("MapVariables", variables.values.toSeq.map(_(quantity)))

      })) >>
      DynamicalSystem.flatten[D]("FlattenVariables")

  protected val output_mapping: Layer[Seq[Output[D]], Seq[Output[D]]] =
    dtflearn.seq_layer[Output[D], Output[D]](name ="CombinedOutput[D]s", system_outputs)

  val model_architecture: Layer[Seq[Output[D]], Seq[Output[D]]] =
    dtflearn.combined_layer("CombineOutput[D]sAndVars", Seq(output_mapping, system_variables_mapping)) >>
      DynamicalSystem.flatten("FlattenModelOutput[D]s")

  protected val system_loss: Layer[(Seq[Output[D]], Seq[Output[D]]), Output[D]] =
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
    ev: Estimator.SupportedInferInput[
      Seq[Output[D]], Seq[Tensor[D]], Seq[Tensor[D]],
      Seq[Output[D]], Seq[Output[D]]]): DynamicalSystem.Model[D] = {

    val model = dtflearn.model[
      Seq[Output[D]], Seq[Output[D]], Seq[Output[D]], Seq[Output[D]],
      D, Seq[Tensor[D]], Seq[D], Seq[Shape], Seq[Tensor[D]], Seq[D], Seq[Shape],
      Seq[Tensor[D]], Seq[Tensor[D]]
      ](
      dtfdata.supervised_dataset.collect(data).map(p => (p._1, p._2)),
      model_architecture,
      (Seq.fill(quantities.size)(input._1), Seq.fill(quantities.size)(input._2)),
      target,
      dtflearn.identity[Seq[Output[D]]]("ID"),
      system_loss, trainConfig,
      data_processing, inMemory,
      graphInstance, Some(data_handles))

    model.train()

    DynamicalSystem.model(
      model,
      quantities.keys.toSeq,
      system_variables.flatMap(_.keys.toSeq)
    )
  }

}


object DynamicalSystem {

  protected case class ObservationalError[D: TF: IsNotQuantized](
    override val name: String,
    error_measure: Layer[(Output[D], Output[D]), Output[D]]) extends
    Layer[(Seq[Output[D]], Seq[Output[D]]), Output[D]](name) {

    override val layerType: String = "ObservationalError"

    override def forwardWithoutContext(input: (Seq[Output[D]], Seq[Output[D]]))(implicit mode: Mode): Output[D] =
      input._1.zip(input._2).map(c => error_measure.forward(c._1, c._2)).reduceLeft(_.add(_))
  }

  protected case class Model[D : TF: IsNotQuantized](
    tfModel: TFModel[
      Seq[Output[D]], Seq[Output[D]], Seq[Output[D]], Seq[Output[D]],
      D, Seq[Tensor[D]], Seq[D], Seq[Shape], Seq[Tensor[D]], Seq[D], Seq[Shape],
      Seq[Tensor[D]], Seq[Tensor[D]]],
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

  def combine_tuple[D: TF: IsNotQuantized]: String => Layer[(Seq[Output[D]], Seq[Output[D]]), Seq[Output[D]]] =
    (name: String) => new Layer[(Seq[Output[D]], Seq[Output[D]]), Seq[Output[D]]](name) {
      override val layerType: String = "CombineSeqTuple"

      override def forwardWithoutContext(input: (Seq[Output[D]], Seq[Output[D]]))(implicit mode: Mode): Seq[Output[D]] =
        input._1 ++ input._2
    }

  def flatten[D: TF: IsNotQuantized](name: String): Layer[Seq[Seq[Output[D]]], Seq[Output[D]]] =
    new Layer[Seq[Seq[Output[D]]], Seq[Output[D]]](name) {
      override val layerType: String = "FlattenSeq"

      override def forwardWithoutContext(input: Seq[Seq[Output[D]]])(implicit mode: Mode): Seq[Output[D]] = input.flatten
    }

  def apply[D : TF: IsNotQuantized: IsFloat32OrFloat64](
    quantities: Map[String, Layer[Output[D], Output[D]]],
    dynamics: Seq[DifferentialOperator[Output[D], Output[D]]],
    input: (D, Shape),
    target: (Seq[D], Seq[Shape]),
    data_loss: Layer[(Output[D], Output[D]), Output[D]],
    quadrature_nodes: Tensor[D],
    weights: Tensor[D],
    loss_weightage: Tensor[D],
    graphInstance: Option[Graph] = None)(
    implicit
    evStructureI: NestedStructure.Aux[Seq[Output[D]], Seq[Tensor[D]], Seq[D], Seq[Shape]],
    evStructure: NestedStructure.Aux[
      (Seq[Output[D]], Seq[Output[D]]),
      (Seq[Tensor[D]], Seq[Tensor[D]]),
      (Seq[D], Seq[D]), (Seq[Shape], Seq[Shape])]): DynamicalSystem[D] =
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