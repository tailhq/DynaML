package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GaussianProcessMixture}
import io.github.mandar2812.dynaml.pipes.DataPipe

import scala.reflect.ClassTag

/**
  * Constructs a gaussian process mixture model
  * from a single [[AbstractGPRegressionModel]] instance.
  * @tparam T The type of the GP training data
  * @tparam I The index set/input domain of the GP model.
  * @author mandar2812 date 15/06/2017.
  * */
class ProbGPMixtureMachine[T, I: ClassTag](
  model: AbstractGPRegressionModel[T, I]) extends
  AbstractCSA[AbstractGPRegressionModel[T, I], GaussianProcessMixture[I]](model) {

  private var policy: String = "CSA"

  private var baselinePolicy: String = "max"

  def _policy = policy

  def setPolicy(p: String): this.type = {
    if(p == "CSA" || p == "Coupled Simulated Annealing")
      policy = "CSA"
    else
      policy = "GS"

    this
  }

  def setBaseLinePolicy(p: String): this.type = {

    if(p == "avg" || p == "mean" || p == "average")
      baselinePolicy = "mean"
    else if(p == "min")
      baselinePolicy = "min"
    else if(p == "max")
      baselinePolicy = "max"
    else
      baselinePolicy = "mean"

    this
  }

  private def calculateEnergyLandscape(initialConfig: Map[String, Double], options: Map[String, String]) =
    if(policy == "CSA") performCSA(initialConfig, options)
    else getEnergyLandscape(initialConfig, options, meanFieldPrior)

  private def modelProbabilities = DataPipe(ProbGPCommMachine.calculateModelWeightsSigmoid(baselinePolicy) _)

  override def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String]) = {

    //Find out the blocked hyper parameters and their values
    val blockedHypParams = system.covariance.blocked_hyper_parameters ++ system.noiseModel.blocked_hyper_parameters

    val (kernelPipe, noisePipe) = (system.covariance.asPipe, system.noiseModel.asPipe)

    val blockedState = system._current_state.filterKeys(blockedHypParams.contains)

    val energyLandscape = calculateEnergyLandscape(initialConfig, options)

    val data = system.data

    //Calculate the weights of each configuration
    val (weights, models) = modelProbabilities(energyLandscape).map(c => {

      val model_state = c._2 ++ blockedState

      implicit val transform = DataPipe(system.dataAsSeq _)

      val model = AbstractGPRegressionModel(
        kernelPipe(model_state), noisePipe(model_state),
        system.mean)(
        data, system.npoints)

      //Persist the model inference primitives to memory.
      model.persist(model_state)

      (c._1, model)
    }).unzip


    val configsAndWeights = modelProbabilities(energyLandscape).map(c => (c._1, c._2 ++ blockedState))

    logger.info("===============================================")
    logger.info("Constructing Gaussian Process Mixture")

    logger.info("Number of model instances = "+weights.length)
    logger.info("--------------------------------------")
    logger.info(
      "Calculated model probabilities/weights are \n"+
        configsAndWeights.map(wc =>
          "\nConfiguration: \n"+
            GlobalOptimizer.prettyPrint(wc._2)+
            "\nProbability = "+wc._1+"\n"
        ).reduceLeft((a, b) => a++b)
    )
    logger.info("--------------------------------------")



    (new GaussianProcessMixture[I](models, DenseVector(weights.toArray)), Map())
  }
}
