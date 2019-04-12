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
package io.github.mandar2812.dynaml.models

import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.evaluation.Performance
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, MetaPipe}
import io.github.mandar2812.dynaml.tensorflow.data.{DataSet, TFDataSet}
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import ammonite.ops._
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.learn.hooks.Evaluator
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.{Function, Output}
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api._
import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}

/**
  * <h4>Hyper-parameter based Tensorflow Model</h4>
  *
  * @tparam IT The type representing input tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam In The type representing symbolic tensors of the input patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam ID The input pattern's TensorFlow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam IS The type of the input pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  * @tparam ArchOut The type of the symbolic tensor returned by the neural architecture,
  *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
  * @tparam TT The type representing target/label tensors,
  *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
  * @tparam Out The type representing symbolic tensors of the target patterns,
  *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
  * @tparam TD The target pattern's TensorFlow data type,
  *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
  * @tparam TS The type of the target pattern's shape,
  *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
  *
  * @param modelFunction A DynaML [[MetaPipe]] (Curried Function) which takes the
  *                      hyper-parameters as input, returns a [[DataPipe]] which
  *                      outputs a [[TFModel]] given some training data.
  *
  * @param hyp_params The string identifiers of the hyper-parameters.
  *
  * @param training_data Training data, as a DynaML [[DataSet]].
  *
  * @param fitness_function A [[DataPipe2]] which takes a prediction-target tuple and outputs a
  *                         performance metric or fitness value. Ideally this value
  *                         should follow the "Lesser is better" paradigm.
  *
  * @param validation_data A validation data set, defaults to [[None]]. This data set
  *                        need not be specified if the [[data_split_func]] is defined.
  *
  * @param data_split_func A [[DataPipe]] which splits the [[training_data]] into train and
  *                        validation splits, defaults to [[None]]. This need not be specified
  *                        if [[validation_data]] is defined.
  *
  * */
class TunableTFModel[
  Pattern,
  In,
  Out,
  ArchOut,
  Loss: TF: IsFloatOrDouble,
  IT,
  ID,
  IS,
  TT,
  TD,
  TS,
  ITT,
  IDD,
  ISS
](val modelFunction: TunableTFModel.ModelFunc[
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ],
  val modelConfigFunc: TunableTFModel.ModelConfigFunc[In, Out],
  val hyp_params: Seq[String],
  protected val training_data: DataSet[Pattern],
  val tf_data_handle_ops: TFModel.TFDataHandleOps[
    Pattern,
    IT,
    TT,
    ITT,
    In,
    Out
  ],
  val fitness_functions: Seq[DataPipe2[ArchOut, Out, Output[Float]]],
  val fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
    DataPipe[Seq[Tensor[Float]], Double](m =>
        m.map(_.scalar.toDouble).sum / m.length),
  protected val validation_data: Option[DataSet[Pattern]] = None,
  protected val data_split_func: Option[DataPipe[Pattern, Boolean]] = None)
    extends GloballyOptimizable {

  //Implicit required by the json4s library for reading and writing json
  implicit protected val formats: Formats = DefaultFormats

  override protected var hyper_parameters: List[String] = hyp_params.toList

  override protected var current_state: TunableTFModel.HyperParams = Map()

  protected def _data_splits: TFDataSet[Pattern] = {

    require(
      validation_data.isDefined || data_split_func.isDefined,
      "If validation data is not explicitly provided, then data_split_func must be defined"
    )

    if (validation_data.isEmpty) training_data.partition(data_split_func.get)
    else TFDataSet(training_data, validation_data.get)
  }

  //Obtain training and validation data splits
  val TFDataSet(train_split, validation_split) = _data_splits

  val (train_data_tf, validation_data_tf): (
    Dataset[(In, Out)],
    Dataset[(In, Out)]
  ) = {

    (
      modelFunction.data_handle(train_split, tf_data_handle_ops),
      modelFunction.data_handle(validation_split, tf_data_handle_ops)
    )
  }

  private val fitness_metrics = fitness_functions.map(fitness_function => {
    Performance[(ArchOut, (In, Out))](
      "Energy",
      DataPipe[(ArchOut, (In, Out)), Output[Float]](
        c => fitness_function(c._1, c._2._2)
      )
    )
  })

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(
    h: TunableTFModel.HyperParams,
    options: Map[String, String]
  ): Double = {

    //Check that all the model hyper-parameters are contained in
    //the input `h`
    require(
      hyp_params.forall(h.contains),
      s"All hyper-parameters: [$hyp_params] of the model, must be contained in the input `h` when calling energy(h)"
    )

    //Set the current state to `h`
    current_state = h

    //Get the model instance.
    val model_instance = modelFunction(h)
    val train_config   = modelConfigFunc(h)

    //Compute the model fitness, guard against weird exceptions
    val (fitness, comment) = try {
      //Train the model instance
      model_instance.train(
        modelFunction._build_ops(train_data_tf, train_config.data_processing),
        train_config
      )

      //Dont shuffle and repeat the data set when performing validation
      val computed_energy = fitness_to_scalar(
        model_instance
          .evaluate(
            modelFunction._build_ops(
              validation_data_tf,
              train_config.data_processing.copy(shuffleBuffer = 0, repeat = 0)
            ),
            fitness_metrics,
            maxSteps = math
              .ceil(
                validation_split.size.toDouble / train_config.data_processing.batchSize
              )
              .toLong,
            saveSummaries = true,
            name = null
          )
      )

      //If all goes well, return the fitness and no comment.
      (computed_energy, None)
    } catch {
      case e: java.lang.IllegalStateException =>
        (Double.PositiveInfinity, Some(e.getMessage))
      case e: Throwable =>
        e.printStackTrace()
        (Double.PositiveInfinity, Some(e.getMessage))
    } finally {
      model_instance.close()
    }

    //Append the model fitness to the hyper-parameter configuration
    val hyp_config_json = write_json(
      h ++ Map("energy" -> fitness, "comment" -> comment.getOrElse(""))
    )

    //Write the configuration along with its fitness into the model
    //instance's summary directory
    write.append(train_config.summaryDir / "state.json", hyp_config_json+"\n")

    //Return the model fitness.
    fitness
  }

  def train_model(
    hyper_params: TunableTFModel.HyperParams,
    trainConfig: Option[TFModel.Config[In, Out]] = None,
    evaluation_metrics: Option[
      Seq[(String, DataPipe2[ArchOut, Out, Output[Float]])]
    ] = None,
    stepTrigger: Option[Int] = None
  ): TFModel[In, Out, ArchOut, Loss, IT, ID, IS, TT, TD, TS, ITT, IDD, ISS] = {

    val model_instance = modelFunction(hyper_params)

    val training_configuration = trainConfig match {
      case None         => modelConfigFunc(hyper_params)
      case Some(config) => config
    }

    if (evaluation_metrics.isDefined) {

      val evalHook = modelFunction._eval_hook(
        Seq(
          (
            "validation",
            modelFunction._build_ops(
              validation_data_tf,
              training_configuration.data_processing
                .copy(shuffleBuffer = 0, repeat = 0)
            )
          )
        ),
        evaluation_metrics.get,
        training_configuration.summaryDir,
        stepTrigger.getOrElse(100)
      )

      model_instance.train(
        modelFunction
          ._build_ops(train_data_tf, training_configuration.data_processing),
        training_configuration.copy(
          trainHooks = training_configuration.trainHooks.map(_ ++ Set(evalHook))
        )
      )
    } else {
      model_instance.train(
        modelFunction._build_ops(
          modelFunction.data_handle(training_data, tf_data_handle_ops),
          training_configuration.data_processing
        ),
        training_configuration
      )
    }

    model_instance
  }

}

object TunableTFModel {

  type HyperParams = Map[String, Double]

  /**
    * Type-alias for "Model Functions".
    *
    * Model Functions take hyper-parameters as input
    * and return an instantiated TensorFlow Model [[TFModel]].
    *
    * */
  type ModelFunc[
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ] =
    ModelFunction[In, Out, ArchOut, Loss, IT, ID, IS, TT, TD, TS, ITT, IDD, ISS]

  type ModelConfigFunc[In, Out] = DataPipe[HyperParams, TFModel.Config[In, Out]]

  /**
    * A configurable Pipeline for creating Tensorflow Model configurations
    * from hyper-parameter maps.
    *
    * @param summaryDir A Pipe ([[DataPipe]]) which generates the summary directory
    *                   given some hyper-parameters.
    *
    * @param data_processing A Pipe which creates the appropriate data handle
    *                        transformations such as shuffle, mini-batching etc.
    *
    * @param optimizer Get the optimization algorithm based on the hyper-parameters.
    *                  Note that if the tunable model using this (i.e. [[TunableTFModel]])
    *                  uses previously stored checkpoints for some hyper-parameters, the mapping
    *                  functions optimizer, and summaryDir must be <i>one is to one</i>.
    *                  Defaults to Adam optimzer with learning rate 0.01.
    *
    * @param stopCriteria Get the stopping criterion for hyper-parameters.
    *
    * @param trainHooks Optional training hooks.
    *
    * */
  class ModelConfigFunction[In, Out](
    summaryDir: DataPipe[HyperParams, Path],
    data_processing: DataPipe[HyperParams, TFModel.Ops[In, Out]] =
      DataPipe[HyperParams, TFModel.Ops[In, Out]](_ =>
          TFModel.data_ops(10000, 16, 10)),
    optimizer: DataPipe[HyperParams, Optimizer] =
      DataPipe[HyperParams, Optimizer](_ => tf.train.Adam(0.01f)),
    stopCriteria: DataPipe[HyperParams, StopCriteria] =
      DataPipe[HyperParams, StopCriteria](_ =>
          dtflearn.rel_loss_change_stop(0.05, 100000)),
    trainHooks: DataPipe[HyperParams, Option[Set[Hook]]] =
      DataPipe[HyperParams, Option[Set[Hook]]](_ => None))
      extends DataPipe[HyperParams, TFModel.Config[In, Out]] {

    override def run(data: HyperParams): TFModel.Config[In, Out] =
      TFModel.trainConfig(
        summaryDir(data),
        data_processing(data),
        optimizer(data),
        stopCriteria(data),
        trainHooks(data)
      )
  }

  object ModelConfigFunction {

    def apply[In, Out](
      summaryDir: DataPipe[HyperParams, Path],
      data_processing: DataPipe[HyperParams, TFModel.Ops[In, Out]] =
        DataPipe[HyperParams, TFModel.Ops[In, Out]](_ =>
            TFModel.data_ops(10000, 16, 10)),
      optimizer: DataPipe[HyperParams, Optimizer] =
        DataPipe[HyperParams, Optimizer](_ => tf.train.Adam(0.01f)),
      stopCriteria: DataPipe[HyperParams, StopCriteria] =
        DataPipe[HyperParams, StopCriteria](_ =>
            dtflearn.rel_loss_change_stop(0.05, 100000)),
      trainHooks: DataPipe[HyperParams, Option[Set[Hook]]] =
        DataPipe[HyperParams, Option[Set[Hook]]](_ => None)
    ) = new ModelConfigFunction(
      summaryDir,
      data_processing,
      optimizer,
      stopCriteria,
      trainHooks
    )
  }

  /**
    * A Model Function creates a TensorFlow model instance
    * given some hyper-parameters.
    *
    * @param generator A model generator pipeline.
    * @param input The input data-type and shape
    * @param target The data-type and shape of the targets.
    *
    * */
  class ModelFunction[
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ](val generator: DataPipe[
      HyperParams,
      TFModel[In, Out, ArchOut, Loss, IT, ID, IS, TT, TD, TS, ITT, IDD, ISS]
    ],
    val input: (ID, IS),
    val target: (TD, TS)
  )(
    implicit
    evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
    evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
    evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
    evOutputStructure: OutputStructure[(In, Out)],
    evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
    evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
    evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)]) {

    def apply(
      config: HyperParams
    ): TFModel[In, Out, ArchOut, Loss, IT, ID, IS, TT, TD, TS, ITT, IDD, ISS] =
      generator(config)

    def data_handle[Pattern](
      data: DataSet[Pattern],
      tf_handle_ops: TFModel.TFDataHandleOps[Pattern, IT, TT, ITT, In, Out]
    ): Dataset[(In, Out)] = {
      TFModel.data._get_tf_data[Pattern, IT, ID, IS, TT, TD, TS, ITT, In, Out](
        data,
        input,
        target,
        tf_handle_ops = tf_handle_ops,
        createOnlyHandle = true
      )
    }

    def _build_ops(
      tf_dataset: Dataset[(In, Out)],
      data_ops: TFModel.Ops[In, Out]
    ): Dataset[(In, Out)] = {
      TFModel.data._build_ops(tf_dataset, data_ops)
    }

    def _eval_hook(
      datasets: Seq[(String, Dataset[(In, Out)])],
      evaluation_metrics: Seq[(String, DataPipe2[ArchOut, Out, Output[Float]])],
      summary_dir: Path,
      stepTrigger: Int = 100,
      log: Boolean = true
    ): Evaluator[
      In,
      (In, Out),
      Out,
      ArchOut,
      Loss,
      (ArchOut, (In, Out)),
      (ID, TD),
      (IS, TS)
    ] =
      TFModel._eval_hook(
        datasets.map(kv => (kv._1, () => kv._2)),
        evaluation_metrics,
        summary_dir,
        stepTrigger,
        log
      )

  }

  /**
    * <h4>Model Functions</h4>
    *
    * Helpful methods for creating [[ModelFunc]] instances, which
    * are needed for creating [[TunableTFModel]].
    * */
  object ModelFunction {

    val config_to_str: DataPipe[Map[String, Double], String] = DataPipe(
      _.map(c => s"${c._1}_${c._2}").mkString("-")
    )
    val generate_token: DataPipe[String, String] = DataPipe(
      utils.tokenGenerator.generateMD5Token
    )

    val to_token = config_to_str > generate_token

    def get_summary_dir(
      top_dir: Path,
      h: HyperParams,
      create_working_dir: Option[DataPipe[HyperParams, String]] = Some(to_token)
    ): Path = create_working_dir match {
      case None                  => top_dir
      case Some(working_dir_gen) => top_dir / working_dir_gen(h)
    }

    val hyper_params_to_dir: MetaPipe[Path, HyperParams, Path] = MetaPipe(
      top_dir => h => get_summary_dir(top_dir, h)
    )

    /**
      * Create a [[ModelFunc]] from a "loss generator".
      *
      * @param loss_gen A function which takes the [[HyperParams]] and creates
      *                 the Loss function.
      *
      * @param architecture The model architecture.
      *
      * @param input Data type and shape of the model inputs.
      *
      * @param target Data type and shape of the model outputs/training labels.
      *
      *
      * @param get_training_config A data pipe which generates a [[TFModel.Config]]
      *                            object from some hyper-parameter assignment.
      *
      * @param data_processing An instance of type [[TFModel.DataOps]], contains details
      *                        on the data processing pipeline to be applied.
      *
      * @param inMemory Set to true if the model should be entirely in memory. Defaults
      *                 to false.
      *
      * @param existingGraph Defaults to None, set this parameter if the model should
      *                      be created in an existing TensorFlow graph.
      *
      * @param data_handles Defaults to None, set this parameter if you wish to instantiate
      *                     the model input-output handles.
      * */
    def from_loss_generator[
      In,
      Out,
      ArchOut,
      Loss: TF: IsFloatOrDouble,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](loss_gen: HyperParams => Layer[(ArchOut, Out), Output[Loss]],
      architecture: Layer[In, ArchOut],
      input: (ID, IS),
      target: (TD, TS),
      inMemory: Boolean = false,
      existingGraph: Option[Graph] = None,
      data_handles: Option[TFModel.DataHandles[In, Out]] = None
    )(
      implicit
      evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
      evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
      evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
      evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
      evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
      evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
      evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
      evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
      evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
      evOutputToShapeI: OutputToShape.Aux[In, IS],
      evOutputToShapeT: OutputToShape.Aux[Out, TS],
      evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
      evOutputStructure: OutputStructure[(In, Out)],
      evOutputStructureArchOut: OutputStructure[ArchOut],
      evOutputStructureI: OutputStructure[In],
      evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
      evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
      evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
      evOutputToTensorIn: OutputToTensor.Aux[In, IT],
      ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
      // This implicit helps the Scala 2.11 compiler.
      evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
    ): ModelFunc[
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ] = {

      new ModelFunction(
        DataPipe(
          (h: TunableTFModel.HyperParams) =>
            TFModel(
              architecture,
              input,
              target,
              loss_gen(h),
              inMemory,
              existingGraph,
              data_handles
            )
        ),
        input,
        target
      )
    }

    /**
      * Create a [[ModelFunc]] from a "architecture-loss generator".
      *
      * @param arch_loss_gen A function which takes the [[HyperParams]] and creates
      *                      an architecture-loss tuple.
      *
      * @param input Data type and shape of the model inputs.
      *
      * @param target Data type and shape of the model outputs/training labels.
      *
      *
      * @param get_training_config A data pipe which generates a [[TFModel.Config]]
      *                            object from some hyper-parameter assignment.
      *
      * @param data_processing An instance of type [[TFModel.DataOps]], contains details
      *                        on the data processing pipeline to be applied.
      *
      * @param inMemory Set to true if the model should be entirely in memory. Defaults
      *                 to false.
      *
      * @param existingGraph Defaults to None, set this parameter if the model should
      *                      be created in an existing TensorFlow graph.
      *
      * @param data_handles Defaults to None, set this parameter if you wish to instantiate
      *                     the model input-output handles.
      * */
    def from_arch_loss_generator[
      In,
      Out,
      ArchOut,
      Loss: TF: IsFloatOrDouble,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](arch_loss_gen: HyperParams => (Layer[In, ArchOut],
        Layer[(ArchOut, Out), Output[Loss]]),
      input: (ID, IS),
      target: (TD, TS),
      inMemory: Boolean = false,
      existingGraph: Option[Graph] = None,
      data_handles: Option[TFModel.DataHandles[In, Out]] = None
    )(
      implicit
      evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
      evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
      evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
      evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
      evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
      evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
      evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
      evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
      evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
      evOutputToShapeI: OutputToShape.Aux[In, IS],
      evOutputToShapeT: OutputToShape.Aux[Out, TS],
      evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
      evOutputStructure: OutputStructure[(In, Out)],
      evOutputStructureArchOut: OutputStructure[ArchOut],
      evOutputStructureI: OutputStructure[In],
      evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
      evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
      evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
      evOutputToTensorIn: OutputToTensor.Aux[In, IT],
      ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
      // This implicit helps the Scala 2.11 compiler.
      evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
    ): ModelFunc[
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ] = {

      new ModelFunction(
        DataPipe(
          (h: TunableTFModel.HyperParams) => {

            val (architecture, loss) = arch_loss_gen(h)

            TFModel(
              architecture,
              input,
              target,
              loss,
              inMemory,
              existingGraph,
              data_handles
            )
          }
        ),
        input,
        target
      )

    }

    /**
      * Create a [[ModelFunc]] from a "architecture generator".
      *
      * @param arch_generator A function which takes the [[HyperParams]] and creates
      *                       a neural architecture.
      *
      * @param input Data type and shape of the model inputs.
      *
      * @param target Data type and shape of the model outputs/training labels.
      *
      *
      * @param get_training_config A data pipe which generates a [[TFModel.Config]]
      *                            object from some hyper-parameter assignment.
      *
      * @param data_processing An instance of type [[TFModel.DataOps]], contains details
      *                        on the data processing pipeline to be applied.
      *
      * @param inMemory Set to true if the model should be entirely in memory. Defaults
      *                 to false.
      *
      * @param existingGraph Defaults to None, set this parameter if the model should
      *                      be created in an existing TensorFlow graph.
      *
      * @param data_handles Defaults to None, set this parameter if you wish to instantiate
      *                     the model input-output handles.
      * */
    def from_arch_generator[
      In,
      Out,
      ArchOut,
      Loss: TF: IsFloatOrDouble,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](arch_generator: HyperParams => Layer[In, ArchOut],
      loss: Layer[(ArchOut, Out), Output[Loss]],
      input: (ID, IS),
      target: (TD, TS),
      inMemory: Boolean = false,
      existingGraph: Option[Graph] = None,
      data_handles: Option[TFModel.DataHandles[In, Out]] = None
    )(
      implicit
      evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
      evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
      evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
      evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
      evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
      evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
      evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
      evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
      evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
      evOutputToShapeI: OutputToShape.Aux[In, IS],
      evOutputToShapeT: OutputToShape.Aux[Out, TS],
      evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
      evOutputStructure: OutputStructure[(In, Out)],
      evOutputStructureArchOut: OutputStructure[ArchOut],
      evOutputStructureI: OutputStructure[In],
      evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
      evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
      evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
      evOutputToTensorIn: OutputToTensor.Aux[In, IT],
      ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
      // This implicit helps the Scala 2.11 compiler.
      evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
    ): ModelFunc[
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ] = {

      new ModelFunction(
        DataPipe(
          (h: TunableTFModel.HyperParams) => {

            val architecture = arch_generator(h)

            TFModel(
              architecture,
              input,
              target,
              loss,
              inMemory,
              existingGraph,
              data_handles
            )
          }
        ),
        input,
        target
      )
    }

  }

  def apply[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss: TF: IsFloatOrDouble,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ](loss_func_gen: HyperParams => Layer[(ArchOut, Out), Output[Loss]],
    hyp: List[String],
    training_data: DataSet[Pattern],
    tf_handle_ops: TFModel.TFDataHandleOps[Pattern, IT, TT, ITT, In, Out],
    fitness_functions: Seq[DataPipe2[ArchOut, Out, Output[Float]]],
    architecture: Layer[In, ArchOut],
    input: (ID, IS),
    target: (TD, TS),
    get_training_config: ModelConfigFunc[In, Out],
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](m =>
          m.map(_.scalar.toDouble).sum / m.length),
    validation_data: Option[DataSet[Pattern]] = None,
    data_split_func: Option[DataPipe[Pattern, Boolean]] = None,
    inMemory: Boolean = false,
    existingGraph: Option[Graph] = None,
    data_handles: Option[TFModel.DataHandles[In, Out]] = None
  )(
    implicit
    evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
    evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
    evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
    evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
    evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
    evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
    evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
    evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
    evOutputToShapeI: OutputToShape.Aux[In, IS],
    evOutputToShapeT: OutputToShape.Aux[Out, TS],
    evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
    evOutputStructure: OutputStructure[(In, Out)],
    evOutputStructureArchOut: OutputStructure[ArchOut],
    evOutputStructureI: OutputStructure[In],
    evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
    evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
    evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
  ): TunableTFModel[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ] = {

    val modelFunc = ModelFunction.from_loss_generator(
      loss_func_gen,
      architecture,
      input,
      target,
      inMemory,
      existingGraph,
      data_handles
    )

    new TunableTFModel[
      Pattern,
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](
      modelFunc,
      get_training_config,
      hyp,
      training_data,
      tf_handle_ops,
      fitness_functions,
      fitness_to_scalar,
      validation_data,
      data_split_func
    )

  }

  def apply[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss: TF: IsFloatOrDouble,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ](arch_loss_gen: HyperParams => (Layer[In, ArchOut],
      Layer[(ArchOut, Out), Output[Loss]]),
    hyp: List[String],
    training_data: DataSet[Pattern],
    tf_handle_ops: TFModel.TFDataHandleOps[Pattern, IT, TT, ITT, In, Out],
    fitness_functions: Seq[DataPipe2[ArchOut, Out, Output[Float]]],
    input: (ID, IS),
    target: (TD, TS),
    get_training_config: ModelConfigFunc[In, Out],
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double],
    validation_data: Option[DataSet[Pattern]],
    data_split_func: Option[DataPipe[Pattern, Boolean]],
    inMemory: Boolean,
    existingGraph: Option[Graph],
    data_handles: Option[TFModel.DataHandles[In, Out]]
  )(
    implicit
    evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
    evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
    evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
    evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
    evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
    evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
    evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
    evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
    evOutputToShapeI: OutputToShape.Aux[In, IS],
    evOutputToShapeT: OutputToShape.Aux[Out, TS],
    evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
    evOutputStructure: OutputStructure[(In, Out)],
    evOutputStructureArchOut: OutputStructure[ArchOut],
    evOutputStructureI: OutputStructure[In],
    evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
    evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
    evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
  ): TunableTFModel[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ] = {

    val modelFunc = ModelFunction.from_arch_loss_generator(
      arch_loss_gen,
      input,
      target,
      inMemory,
      existingGraph,
      data_handles
    )

    new TunableTFModel[
      Pattern,
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](
      modelFunc,
      get_training_config,
      hyp,
      training_data,
      tf_handle_ops,
      fitness_functions,
      fitness_to_scalar,
      validation_data,
      data_split_func
    )

  }

  def apply[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss: TF: IsFloatOrDouble,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ](arch_gen: HyperParams => Layer[In, ArchOut],
    loss: Layer[(ArchOut, Out), Output[Loss]],
    hyp: List[String],
    training_data: DataSet[Pattern],
    tf_handle_ops: TFModel.TFDataHandleOps[Pattern, IT, TT, ITT, In, Out],
    fitness_functions: Seq[DataPipe2[ArchOut, Out, Output[Float]]],
    input: (ID, IS),
    target: (TD, TS),
    get_training_config: ModelConfigFunc[In, Out],
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double],
    validation_data: Option[DataSet[Pattern]],
    data_split_func: Option[DataPipe[Pattern, Boolean]],
    inMemory: Boolean,
    existingGraph: Option[Graph],
    data_handles: Option[TFModel.DataHandles[In, Out]]
  )(
    implicit
    evDataTypeToOutputI: DataTypeToOutput.Aux[ID, In],
    evDataTypeToOutputT: DataTypeToOutput.Aux[TD, Out],
    evTensorToOutput: TensorToOutput.Aux[(IT, TT), (In, Out)],
    evTensorToDataType: TensorToDataType.Aux[(IT, TT), (ID, TD)],
    evTensorToShape: TensorToShape.Aux[(IT, TT), (IS, TS)],
    evOutputToDataTypeI: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeT: OutputToDataType.Aux[Out, TD],
    evOutputToDataType: OutputToDataType.Aux[(In, Out), (ID, TD)],
    evDataTypeToShape: DataTypeToShape.Aux[(ID, TD), (IS, TS)],
    evOutputToShapeI: OutputToShape.Aux[In, IS],
    evOutputToShapeT: OutputToShape.Aux[Out, TS],
    evOutputToShape: OutputToShape.Aux[(In, Out), (IS, TS)],
    evOutputStructure: OutputStructure[(In, Out)],
    evOutputStructureArchOut: OutputStructure[ArchOut],
    evOutputStructureI: OutputStructure[In],
    evOutputToDataTypeOut: OutputToDataType.Aux[ArchOut, IDD],
    evOutputToShapeOut: OutputToShape.Aux[ArchOut, ISS],
    evOutputToTensorOut: OutputToTensor.Aux[ArchOut, ITT],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, IT, ITT],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, ArchOut), (IT, ITT)]
  ): TunableTFModel[
    Pattern,
    In,
    Out,
    ArchOut,
    Loss,
    IT,
    ID,
    IS,
    TT,
    TD,
    TS,
    ITT,
    IDD,
    ISS
  ] = {

    val modelFunc = ModelFunction.from_arch_generator(
      arch_gen,
      loss,
      input,
      target,
      inMemory,
      existingGraph,
      data_handles
    )

    new TunableTFModel[
      Pattern,
      In,
      Out,
      ArchOut,
      Loss,
      IT,
      ID,
      IS,
      TT,
      TD,
      TS,
      ITT,
      IDD,
      ISS
    ](
      modelFunc,
      get_training_config,
      hyp,
      training_data,
      tf_handle_ops,
      fitness_functions,
      fitness_to_scalar,
      validation_data,
      data_split_func
    )

  }

}
