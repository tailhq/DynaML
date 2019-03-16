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
private[dynaml] class PDESystem(
  val quantities: Layer[Output[Float], Output[Float]],
  val dynamics: DifferentialOperator[Output[Float], Output[Float]],
  val input_shape: Shape,
  val target_shape: Shape,
  val data_loss: Layer[(Output[Float], Output[Float]), Output[Float]],
  quadrature_nodes: Tensor[Float],
  quadrature_weights: Tensor[Float],
  quadrature_loss_weightage: Tensor[Float],
  graphInstance: Option[Graph],
  name: String = "Output") {

  protected val observational_error: Layer[(Output[Float], Output[Float]), Output[Float]] =
    PDESystem.error("ExtObsError", data_loss)

  protected val projection = PDESystem.projectOutputs("ProjectOutputs")

  val system_outputs: Layer[Output[Float], Output[Float]] = quantities

  protected val quadrature: PDEQuadrature[Float] =
    PDEQuadrature(
      "ColocationError",
      dynamics(system_outputs),
      quadrature_nodes, quadrature_weights,
      quadrature_loss_weightage)

  val system_variables: Map[String, DifferentialOperator[Output[Float], Output[Float]]] =
    dynamics.variables

  private val data_handles = (
    tf.learn.Input[Output[Float], FLOAT32, Shape](FLOAT32, Shape(-1) ++ input_shape,  name = "Input"),
    tf.learn.Input[Output[Float], FLOAT32, Shape](FLOAT32, Shape(-1) ++ target_shape, name = "Outputs")
  )

  val system_variables_mapping: Layer[Output[Float], Map[String, Output[Float]]] =
    dtflearn.map_layer("MapVars", system_variables.map(kv => (kv._1, kv._2(system_outputs))))

  protected val output_mapping: Layer[Output[Float], Output[Float]] = system_outputs

  val model_architecture: Layer[Output[Float], PDESystem.ModelOutputs] =
    dtflearn.bifurcation_layer("CombineOutputsAndVars", output_mapping, system_variables_mapping)

  protected val system_loss: Layer[(PDESystem.ModelOutputs, Output[Float]), Output[Float]] =
    projection >>
      observational_error >>
      quadrature >>
      tf.learn.Mean[Float]("Loss/Mean") >>
      tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")

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
    data: SupervisedDataSet[Tensor[Float], Tensor[Float]],
    trainConfig: TFModel.Config,
    data_processing: TFModel.Ops = TFModel.data_ops(10000, 16, 10),
    inMemory: Boolean = false,
    concatOpI: Option[DataPipe[Iterable[Tensor[Float]], Tensor[Float]]] = None,
    concatOpT: Option[DataPipe[Iterable[Tensor[Float]], Tensor[Float]]] = None,
    concatOpO: Option[DataPipe[Iterable[PDESystem.ModelOutputsT], PDESystem.ModelOutputsT]] = None)
  : PDESystem.Model = {

    val model = dtflearn.model[
      Output[Float], Output[Float], PDESystem.ModelOutputs, Float,
      Tensor[Float], FLOAT32, Shape,
      Tensor[Float], FLOAT32, Shape,
      PDESystem.ModelOutputsT,
      (FLOAT32, Map[String, FLOAT32]),
      (Shape, Map[String, Shape])](
      model_architecture,
      (FLOAT32, input_shape),
      (FLOAT32, target_shape),
      system_loss, trainConfig,
      data_processing, inMemory,
      graphInstance, Some(data_handles),
      concatOpI, concatOpT, concatOpO)

    model.train(data)

    PDESystem.model(model, name, system_variables.keys.toSeq)
  }

}


object PDESystem {

  type ModelOutputs = (Output[Float], Map[String, Output[Float]])
  type ModelOutputsT = (Tensor[Float], Map[String, Tensor[Float]])

  protected case class ObservationalError(
    override val name: String,
    error_measure: Layer[(Output[Float], Output[Float]), Output[Float]]) extends
    Layer[(Output[Float], Output[Float]), Output[Float]](name) {

    override val layerType: String = "ObservationalError"

    override def forwardWithoutContext(
      input: (Output[Float], Output[Float]))(
      implicit mode: Mode): Output[Float] = error_measure.forwardWithoutContext(input._1, input._2)
  }

  protected case class ProjectOutputs(override val name: String)
    extends Layer[
      ((Output[Float], Map[String, Output[Float]]), Output[Float]),
      (Output[Float], Output[Float])] {
    override val layerType: String = "ProjectOutputs"

    override def forwardWithoutContext(
      input: ((Output[Float], Map[String, Output[Float]]), Output[Float]))(
      implicit mode: Mode): (Output[Float], Output[Float]) =
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
      Output[Float], Output[Float], PDESystem.ModelOutputs, Float,
      Tensor[Float], FLOAT32, Shape,
      Tensor[Float], FLOAT32, Shape,
      PDESystem.ModelOutputsT,
      (FLOAT32, Map[String, FLOAT32]),
      (Shape, Map[String, Shape])],
    outputs: String,
    variables: Seq[String]) {

    private val model_quantities = Seq(outputs) ++ variables

    protected def predict(input: Tensor[Float]): Map[String, Tensor[Float]] = {

      val model_preds = tfModel.predict(input)

      Map(outputs -> model_preds._1) ++ model_preds._2
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

  def apply(
    quantities: Layer[Output[Float], Output[Float]],
    dynamics: DifferentialOperator[Output[Float], Output[Float]],
    input: Shape,
    target: Shape,
    data_loss: Layer[(Output[Float], Output[Float]), Output[Float]],
    quadrature_nodes: Tensor[Float],
    weights: Tensor[Float],
    loss_weightage: Tensor[Float],
    graphInstance: Option[Graph] = None): PDESystem =
    new PDESystem(
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