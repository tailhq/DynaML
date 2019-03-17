package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.models.TFModel
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.layers.PDEQuadrature
import io.github.mandar2812.dynaml.tensorflow.data._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._

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
private[dynaml] class PDESystem[T: TF: IsDecimal, U: TF: IsDecimal, L: TF: IsFloatOrDouble](
  val quantities: Layer[Output[T], Output[U]],
  val dynamics: DifferentialOperator[Output[T], Output[U]],
  val input_shape: Shape,
  val target_shape: Shape,
  val data_loss: Layer[(Output[U], Output[U]), Output[L]],
  quadrature_nodes: Tensor[T],
  quadrature_weights: Tensor[U],
  quadrature_loss_weightage: Tensor[U],
  graphInstance: Option[Graph],
  name: String = "Output") {

  protected val observational_error: Layer[(Output[U], Output[U]), Output[L]] =
    PDESystem.error[U, L]("ExtObsError", data_loss)

  protected val projection = PDESystem.projectOutputs[U]("ProjectOutputs")

  val system_outputs: Layer[Output[T], Output[U]] = quantities

  protected val quadrature: PDEQuadrature[T, U, L] =
    PDEQuadrature(
      "ColocationError",
      dynamics(system_outputs),
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)

  val system_variables: Map[String, DifferentialOperator[Output[T], Output[U]]] =
    dynamics.variables

  private val dTypeTag = TF[T]
  private val dTypeTagO = TF[U]

  private val data_handles = (
    tf.learn.Input[Output[T], DataType[T], Shape](dTypeTag.dataType, Shape(-1) ++ input_shape,  name = "Input"),
    tf.learn.Input[Output[U], DataType[U], Shape](dTypeTagO.dataType, Shape(-1) ++ target_shape, name = "Outputs")
  )

  val system_variables_mapping: Layer[Output[T], Map[String, Output[U]]] =
    dtflearn.map_layer("MapVars", system_variables.map(kv => (kv._1, kv._2(system_outputs))))

  val model_architecture: Layer[Output[T], PDESystem.ModelOutputs[U]] =
    dtflearn.bifurcation_layer("CombineOutputsAndVars", system_outputs, system_variables_mapping)

  protected val system_loss: Layer[(PDESystem.ModelOutputs[U], Output[U]), Output[L]] =
    projection >>
      observational_error >>
      quadrature >>
      tf.learn.Mean[L]("Loss/Mean") >>
      tf.learn.ScalarSummary[L]("Loss/Summary", "Loss")

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
    trainConfig: TFModel.Config,
    data_processing: TFModel.Ops = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false,
    concatOpI: Option[DataPipe[Iterable[Tensor[T]], Tensor[T]]] = None,
    concatOpT: Option[DataPipe[Iterable[Tensor[U]], Tensor[U]]] = None,
    concatOpO: Option[DataPipe[Iterable[PDESystem.ModelOutputsT[U]], PDESystem.ModelOutputsT[U]]] = None)
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
      system_loss, trainConfig,
      data_processing, inMemory,
      graphInstance, Some(data_handles),
      concatOpI, concatOpT, concatOpO)

    model.train(data)

    PDESystem.model(model, name, system_variables.keys.toSeq)
  }

}


object PDESystem {

  type ModelOutputs[T]  = (Output[T], Map[String, Output[T]])
  type ModelOutputsT[T] = (Tensor[T], Map[String, Tensor[T]])

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
    name: String = "Output"): PDESystem[T, U, L] =
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
      name
    )
  
}