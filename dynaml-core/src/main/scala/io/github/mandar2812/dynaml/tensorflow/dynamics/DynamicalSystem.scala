package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.PDEQuadrature
import io.github.mandar2812.dynaml.tensorflow.data._
//import org.platanios.tensorflow.api.implicits.helpers.{DataTypeToOutput, OutputToShape}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._

//TODO: Fix issues around system variables, or lack there of
/**
  *
  * <h3>Dynamical Systems</h3>
  * <h4>Forecasting, System Identification and Design</h4>
  *
  * @param quantities A map containing quantity names and their associated function approximations.
  * @param dynamics A system of differential operators, representing the system dynamics
  * @param input_shape The data type and shape of the input tensor.
  * @param target_shape The data types and shapes of the outputs and the latent variables.
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
private[dynaml] class DynamicalSystem(
  val quantities: Map[String, Layer[Output[Float], Output[Float]]],
  val dynamics: Seq[DifferentialOperator[Output[Float], Output[Float]]],
  val input_shape: Shape,
  val target_shape: Seq[Shape],
  val data_loss: Layer[(Output[Float], Output[Float]), Output[Float]],
  quadrature_nodes: Tensor[Float],
  quadrature_weights: Tensor[Float],
  quadrature_loss_weightage: Tensor[Float],
  graphInstance: Option[Graph]) {

  protected val observational_error: Layer[(Seq[Output[Float]], Seq[Output[Float]]), Output[Float]] =
    DynamicalSystem.error("ExtObsError", data_loss)

  protected val projection = DynamicalSystem.projectOutputs("ProjectOutputs")

  val system_outputs: Seq[Layer[Output[Float], Output[Float]]] = quantities.values.toSeq

  protected val quadrature: PDEQuadrature[Float] =
    PDEQuadrature(
      "ColocationError",
      dynamics.zip(system_outputs).map(c => c._1(c._2)),
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)

  val system_variables: Seq[Map[String, DifferentialOperator[Output[Float], Output[Float]]]] =
    dynamics.map(_.variables)

  private val data_handles = (
    tf.learn.Input[Seq[Output[Float]], Seq[FLOAT32], Seq[Shape]](
      Seq.fill(quantities.toSeq.length)(FLOAT32),
      Seq.fill(quantities.toSeq.length)(Shape(-1) ++ input_shape),
      "Input"),
    tf.learn.Input[Seq[Output[Float]], Seq[FLOAT32], Seq[Shape]](
      target_shape.map(_ => FLOAT32),
      target_shape.map(s => Shape(-1) ++ s),
      "Outputs")
  )

  val system_variables_mapping: Layer[Seq[Output[Float]], Seq[Map[String, Output[Float]]]] =
    dtflearn.seq_layer[Output[Float], Map[String, Output[Float]]](
      "SystemVariables",
      system_variables.zip(system_outputs).map(variablesAndQuantities => {

        val (variables, quantity) = variablesAndQuantities

        dtflearn.map_layer("MapVars", variables.map(kv => (kv._1, kv._2(quantity))))

      }))

  protected val output_mapping: Layer[Seq[Output[Float]], Seq[Output[Float]]] =
    dtflearn.seq_layer[Output[Float], Output[Float]](name ="CombinedOutputs", system_outputs)

  val model_architecture: Layer[Seq[Output[Float]], DynamicalSystem.ModelOutputs] =
    dtflearn.bifurcation_layer("CombineOutputsAndVars", output_mapping, system_variables_mapping)

  protected val system_loss: Layer[(DynamicalSystem.ModelOutputs, Seq[Output[Float]]), Output[Float]] =
    projection >>
      observational_error >>
      quadrature >>
      tf.learn.Mean[Float]("Loss/Mean") >>
      tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")

  def solve(
    data: Seq[SupervisedDataSet[Tensor[Float], Tensor[Float]]],
    trainConfig: TFModel.Config,
    data_processing: TFModel.Ops = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false): DynamicalSystem.Model = {

    val model = dtflearn.model[
      Seq[Output[Float]], Seq[Output[Float]], DynamicalSystem.ModelOutputs, Float,
      Seq[Tensor[Float]], Seq[FLOAT32], Seq[Shape],
      Seq[Tensor[Float]], Seq[FLOAT32], Seq[Shape],
      DynamicalSystem.ModelOutputsT,
      (Seq[FLOAT32], Seq[Map[String, FLOAT32]]),
      (Seq[Shape], Seq[Map[String, Shape]])](
      dtfdata.supervised_dataset.collect(data),
      model_architecture,
      (Seq.fill(quantities.size)(FLOAT32), Seq.fill(quantities.size)(input_shape)),
      (Seq.fill(quantities.size)(FLOAT32), target_shape),
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

  type ModelOutputs  = (Seq[Output[Float]], Seq[Map[String, Output[Float]]])
  type ModelOutputsT = (Seq[Tensor[Float]], Seq[Map[String, Tensor[Float]]])

  protected case class ObservationalError(
    override val name: String,
    error_measure: Layer[(Output[Float], Output[Float]), Output[Float]]) extends
    Layer[(Seq[Output[Float]], Seq[Output[Float]]), Output[Float]](name) {

    override val layerType: String = "ObservationalError"

    override def forwardWithoutContext(
      input: (Seq[Output[Float]], Seq[Output[Float]]))(
      implicit mode: Mode): Output[Float] =
      input._1.zip(input._2)
        .map(c => error_measure.forward(c._1, c._2))
        .reduceLeft(tf.add[Float](_, _))
  }

  protected case class ProjectOutputs(override val name: String)
    extends Layer[
      ((Seq[Output[Float]], Seq[Map[String, Output[Float]]]), Seq[Output[Float]]),
      (Seq[Output[Float]], Seq[Output[Float]])] {
    override val layerType: String = "ProjectOutputs"

    override def forwardWithoutContext(
      input: ((Seq[Output[Float]], Seq[Map[String, Output[Float]]]), Seq[Output[Float]]))(
      implicit mode: Mode): (Seq[Output[Float]], Seq[Output[Float]]) =
      (input._1._1, input._2)
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
  case class Model(
    tfModel: TFModel[
      Seq[Output[Float]], Seq[Output[Float]], ModelOutputs, Float,
      Seq[Tensor[Float]], Seq[FLOAT32], Seq[Shape],
      Seq[Tensor[Float]], Seq[FLOAT32], Seq[Shape],
      ModelOutputsT, (Seq[FLOAT32], Seq[Map[String, FLOAT32]]),
      (Seq[Shape], Seq[Map[String, Shape]])],
    outputs: Seq[String],
    variables: Seq[String]) {

    private val model_quantities = outputs ++ variables

    protected def predict(input: Tensor[Float]): Map[String, Tensor[Float]] = {

      val model_preds = tfModel.predict(Seq.fill[Tensor[Float]](outputs.length)(input))

      model_quantities.zip(model_preds._1).toMap ++ model_preds._2.reduceLeft(_++_)
    }

    def predict(quantities: String*)(input: Tensor[Float]): Seq[Tensor[Float]] = {
      require(
        quantities.forall(model_quantities.contains),
        "Each provided quantity should be in the list of model quantities,"+
          " either as an output or as an inferred variable")

      val outputs = predict(input)

      quantities.map(outputs(_))
    }


  }

  val error: ObservationalError.type      = ObservationalError
  val model: Model.type                   = Model
  val projectOutputs: ProjectOutputs.type = ProjectOutputs

  def combine_tuple: String => Layer[(Seq[Output[Float]], Seq[Output[Float]]), Seq[Output[Float]]] =
    (name: String) => new Layer[(Seq[Output[Float]], Seq[Output[Float]]), Seq[Output[Float]]](name) {
      override val layerType: String = "CombineSeqTuple"

      override def forwardWithoutContext(input: (Seq[Output[Float]], Seq[Output[Float]]))(implicit mode: Mode): Seq[Output[Float]] =
        input._1 ++ input._2
    }

  def flatten(name: String): Layer[Seq[Seq[Output[Float]]], Seq[Output[Float]]] =
    new Layer[Seq[Seq[Output[Float]]], Seq[Output[Float]]](name) {
      override val layerType: String = "FlattenSeq"

      override def forwardWithoutContext(input: Seq[Seq[Output[Float]]])(implicit mode: Mode): Seq[Output[Float]] = input.flatten
    }

  def apply(
    quantities: Map[String, Layer[Output[Float], Output[Float]]],
    dynamics: Seq[DifferentialOperator[Output[Float], Output[Float]]],
    input: Shape,
    target: Seq[Shape],
    data_loss: Layer[(Output[Float], Output[Float]), Output[Float]],
    quadrature_nodes: Tensor[Float],
    weights: Tensor[Float],
    loss_weightage: Tensor[Float],
    graphInstance: Option[Graph] = None): DynamicalSystem =
    new DynamicalSystem(
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