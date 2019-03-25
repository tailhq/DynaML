package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.{PDEQuadrature, Regularization, L2Regularization, L1Regularization}
import io.github.mandar2812.dynaml.tensorflow.data._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._

//TODO: Use variable contexts for accessing system sources/epistemic quantities
/**
  *
  * <h3>Dynamical Systems</h3>
  * <h4>Forecasting, System Identification and Design</h4>
  *
  * @param function A map containing quantity names and their associated function approximations.
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
private[dynaml] class PDESystem[T: TF: IsDecimal, U: TF: IsDecimal, L: TF: IsFloatOrDouble](
  val function: Layer[Output[T], Output[U]],
  val dynamics: DifferentialOperator[Output[T], Output[U]],
  val input_shape: Shape,
  val target_shape: Shape,
  val data_loss: Layer[(Output[U], Output[U]), Output[L]],
  quadrature_nodes: Tensor[T],
  quadrature_weights: Tensor[U],
  quadrature_loss_weightage: Tensor[U],
  graphInstance: Option[Graph],
  val reg_f: Option[Regularization[L]] = None,
  val reg_v: Option[Regularization[L]] = None,
  name: String = "Output",
  system_name: Option[String] = None) {

  protected val observational_error: Layer[(Output[U], Output[U]), Output[L]] =
    PDESystem.error[U, L]("ExtObsError", data_loss)

  protected val projection: Layer[(PDESystem.ModelOutputs[U], Output[U]), (Output[U], Output[U])] =
    PDESystem.projectOutputs[U]("ProjectOutputs")

  val system_outputs: PDESystem.ModelLayer[T, U]   = function
  val system_residuals: PDESystem.ModelLayer[T, U] = dynamics(system_outputs)
  val system_variables: PDESystem.VarMap[T, U]     = dynamics.variables.map(kv => (kv._1, kv._2(system_outputs)))

  private val var_scope =
    DataPipe[String, String](dtfutils.get_scope(system_residuals)) >
      DataPipe[String, String](dtfutils.process_scope)

  val variable_scopes: Seq[String] =
    system_variables
      .values
      .toSeq
      .map(v => s"${var_scope(v.name)}${v.name}")

  protected val quadrature: PDEQuadrature[T, U, L] =
    PDEQuadrature(
      "ColocationError",
      system_residuals,
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)



  private val dTypeTag = TF[T]
  private val dTypeTagO = TF[U]

  private val data_handles = (
    tf.learn.Input[Output[T], DataType[T], Shape](dTypeTag.dataType, Shape(-1) ++ input_shape,  name = "Input"),
    tf.learn.Input[Output[U], DataType[U], Shape](dTypeTagO.dataType, Shape(-1) ++ target_shape, name = "Outputs")
  )

  val system_variables_mapping: Layer[Output[T], Map[String, Output[U]]] =
    dtflearn.scoped_map_layer("MapVars", system_variables, variable_scopes)

  val model_architecture: Layer[Output[T], PDESystem.ModelOutputs[U]] =
    dtflearn.bifurcation_layer(
      if(system_name.isDefined) system_name.get else "CombinedOutputsAndVars",
      system_outputs,
      system_variables_mapping)

  private val primary_loss = projection >> observational_error >> quadrature

  private val regularization_f = PDESystem.regularization(model_architecture, reg_f)
  private val regularization_v = PDESystem.regularization(variable_scopes, reg_v)

  protected val system_loss: Layer[(PDESystem.ModelOutputs[U], Output[U]), Output[L]] =
    primary_loss >>
      regularization_f >>
      regularization_v >>
      tf.learn.Mean[L](name = "Loss/Mean") >>
      tf.learn.ScalarSummary[L](name = "Loss/Summary", tag = "Loss")

  private val tensors_to_symbolic = DataPipe[(Tensor[T], Tensor[U]), (Output[T], Output[U])](
    c => (c._1.toOutput, c._2.toOutput)
  )

  /**
    * Train a neural net based approximation for the
    * dynamical system.
    *
    * @param data Training data, a sequence of supervised/labeled data
    *             sets the length of the sequence must equal the number
    *             of governing equations.
    * @param trainConfig Training configuration, of type [[TFModel.Config]]
    * @param data_processing TensorFlow data operation pipeline, instance of [[TFModel.Ops]]
    * @param inMemory Set to true if model is to be kept entirely in memory, defaults to false.
    * @return A [[PDESystem.Model]] which encapsulates a predictive model of type [[TFModel]]
    * */
  def solve(
    data: SupervisedDataSet[Tensor[T], Tensor[U]],
    trainConfig: TFModel.Config[Output[T], Output[U]],
    tf_handle_ops: TFModel.HandleOps[Tensor[T], Tensor[U], PDESystem.ModelOutputsT[U]] = TFModel.tf_data_ops[Tensor[T], Tensor[U], PDESystem.ModelOutputsT[U]](),
    inMemory: Boolean = false)
  : PDESystem.Model[T, U, L] = {

    val model = dtflearn.model[
      Output[T], Output[U], PDESystem.ModelOutputs[U], L,
      Tensor[T], DataType[T], Shape,
      Tensor[U], DataType[U], Shape,
      PDESystem.ModelOutputsT[U],
      (DataType[U], Map[String, DataType[U]]),
      (Shape, Map[String, Shape])](
      model_architecture,
      (dTypeTag.dataType, input_shape),
      (dTypeTagO.dataType, target_shape),
      system_loss, inMemory,
      graphInstance, Some(data_handles), 
      tf_handle_ops)

    model.train(data, trainConfig)

    PDESystem.model(model, name, system_variables.keys.toSeq)
  }

}


object PDESystem {

  type ModelLayer[T, U] = Layer[Output[T], Output[U]]
  type VarMap[T, U]     = Map[String, ModelLayer[T, U]]
  type ModelOutputs[T]  = (Output[T], Map[String, Output[T]])
  type ModelOutputsT[T] = (Tensor[T], Map[String, Tensor[T]])

  def modify_reg[L: TF: IsFloatOrDouble](
    model_architecture: Layer[_, _],
    reg: Regularization[L]) = reg match {

    case l2reg: L2Regularization[L] =>
      l2reg.copy[L](
        scopes = l2reg.names.map(n => dtfutils.get_scope(model_architecture)(n.split("/").head)))

    case l1reg: L1Regularization[L] =>
      l1reg.copy[L](
        scopes = l1reg.names.map(n => dtfutils.get_scope(model_architecture)(n.split("/").head)))
  }

  def modify_reg[L: TF: IsFloatOrDouble](
    scopes: Seq[String],
    reg: Regularization[L]) = reg match {

    case l2reg: L2Regularization[L] =>
      l2reg.copy[L](scopes = scopes)

    case l1reg: L1Regularization[L] =>
      l1reg.copy[L](scopes = scopes)
  }

  def regularization[L: TF: IsFloatOrDouble](
    s: Seq[String],
    reg: Option[Regularization[L]]): Layer[Output[L], Output[L]] = reg match {
    case None => dtflearn.identity[Output[L]]("Id")
    case Some(r) => modify_reg(s, r)
  }

  def regularization[L: TF: IsFloatOrDouble](
    arch: Layer[_, _],
    reg: Option[Regularization[L]]): Layer[Output[L], Output[L]] = reg match {
    case None => dtflearn.identity[Output[L]]("Id")
    case Some(r) => modify_reg(arch, r)
  }

  protected case class ObservationalError[T: TF: IsDecimal, L: TF: IsFloatOrDouble](
    override val name: String,
    error_measure: Layer[(Output[T], Output[T]), Output[L]]) extends
    Layer[(Output[T], Output[T]), Output[L]](name) {

    override val layerType: String = "ObservationalError"

    override def forwardWithoutContext(
      input: (Output[T], Output[T]))(
      implicit mode: Mode): Output[L] = error_measure.forwardWithoutContext(input._1, input._2)
  }

  protected case class ProjectOutputs[T: TF: IsDecimal](override val name: String)
    extends Layer[(ModelOutputs[T], Output[T]), (Output[T], Output[T])] {

    override val layerType: String = "ProjectOutputs"

    override def forwardWithoutContext(
      input: (ModelOutputs[T], Output[T]))(
      implicit mode: Mode): (Output[T], Output[T]) =
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
  case class Model[T: TF: IsDecimal, U: TF: IsDecimal, L: TF: IsFloatOrDouble](
    tfModel: TFModel[
      Output[T], Output[U], PDESystem.ModelOutputs[U], L,
      Tensor[T], DataType[T], Shape,
      Tensor[U], DataType[U], Shape,
      PDESystem.ModelOutputsT[U],
      (DataType[U], Map[String, DataType[U]]),
      (Shape, Map[String, Shape])],
    outputs: String,
    variables: Seq[String]) {

    private val model_quantities = Seq(outputs) ++ variables

    protected def predict(input: Tensor[T]): Map[String, Tensor[U]] = {

      val model_preds = tfModel.predict(input)

      Map(outputs -> model_preds._1) ++ model_preds._2
    }

    def predict(quantities: String*)(input: Tensor[T]): Seq[Tensor[U]] = {
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

  def apply[T: TF: IsDecimal, U: TF: IsDecimal, L: TF: IsFloatOrDouble](
    quantities: Layer[Output[T], Output[U]],
    dynamics: DifferentialOperator[Output[T], Output[U]],
    input: Shape,
    target: Shape,
    data_loss: Layer[(Output[U], Output[U]), Output[L]],
    quadrature_nodes: Tensor[T],
    weights: Tensor[U],
    loss_weightage: Tensor[U],
    graphInstance: Option[Graph] = None,
    reg_f: Option[Regularization[L]] = None,
    reg_v: Option[Regularization[L]] = None,
    name: String = "Output",
    system_name: Option[String] = None): PDESystem[T, U, L] =
    new PDESystem[T, U, L](
      quantities,
      dynamics,
      input,
      target,
      data_loss,
      quadrature_nodes,
      weights,
      loss_weightage,
      graphInstance,
      reg_f, reg_v,
      name, system_name
    )
  
}