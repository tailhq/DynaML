package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Moments}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, GenContinuousMixtureModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability.ContinuousRVWithDistr
import io.github.mandar2812.dynaml.probability.distributions.HasErrorBars

import scala.reflect.ClassTag

/**
  * @author mandar2812 date 21/06/2017.
  * */
abstract class MixtureMachine[
T, I: ClassTag, Y, YDomain, YDomainVar,
BaseDistr <: ContinuousDistr[YDomain]
  with Moments[YDomain, YDomainVar]
  with HasErrorBars[YDomain],
W1 <: ContinuousRVWithDistr[YDomain, BaseDistr],
BaseProcess <: ContinuousProcessModel[T, I, Y, W1]
  with SecondOrderProcessModel[T, I, Y, Double, DenseMatrix[Double], W1]
  with GloballyOptimizable](model: BaseProcess) extends
  AbstractCSA[BaseProcess, GenContinuousMixtureModel[
      T, I, Y, YDomain, YDomainVar,
      BaseDistr, W1, BaseProcess]](model) {


  val confToModel: DataPipe[Map[String, Double], BaseProcess]

  val mixturePipe: DataPipe2[
    Seq[BaseProcess],
    DenseVector[Double],
    GenContinuousMixtureModel[
      T, I, Y, YDomain, YDomainVar,
      BaseDistr, W1, BaseProcess]]


  protected var policy: String = "CSA"

  protected var baselinePolicy: String = "max"

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

  protected def calculateEnergyLandscape(
    initialConfig: Map[String, Double],
    options: Map[String, String]): Seq[(Double, Map[String, Double])] =
    if(policy == "CSA") performCSA(initialConfig, options)
    else getEnergyLandscape(initialConfig, options, meanFieldPrior)

  protected def modelProbabilities
  : DataPipe[Seq[(Double, Map[String, Double])], Seq[(Double, Map[String, Double])]] =
    DataPipe(ProbGPCommMachine.calculateModelWeightsSigmoid(baselinePolicy))

  override def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String]) = {

    //Find out the blocked hyper parameters and their values
    val blockedHypParams = system._hyper_parameters.filterNot(initialConfig.contains)

    val blockedState = system._current_state.filterKeys(blockedHypParams.contains)

    val energyLandscape = calculateEnergyLandscape(initialConfig, options)


    //Calculate the weights of each configuration
    val (weights, models) = modelProbabilities(energyLandscape).map(c => {

      val model_state = c._2 ++ blockedState

      val model = confToModel(model_state)

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



    (
      mixturePipe(
        models, DenseVector(weights.toArray)
      ),
      models.map(m => {
        val model_id = m.toString.split("\\.").last
        m._current_state.map(c => (model_id+"/"+c._1,c._2))
      }).reduceLeft((m1, m2) => m1++m2)
    )
  }

}

object MixtureMachine {

  def apply[
    T, I: ClassTag, Y, YDomain, YDomainVar,
    BaseDistr <: ContinuousDistr[YDomain] with Moments[YDomain, YDomainVar] with HasErrorBars[YDomain],
    W1 <: ContinuousRVWithDistr[YDomain, BaseDistr],
    BaseProcess <: ContinuousProcessModel[T, I, Y, W1]
      with SecondOrderProcessModel[T, I, Y, Double, DenseMatrix[Double], W1]
      with GloballyOptimizable](model: BaseProcess)(
    confModelPipe: DataPipe[Map[String, Double], BaseProcess],
    mixtPipe: DataPipe2[Seq[BaseProcess], DenseVector[Double], GenContinuousMixtureModel[
      T, I, Y, YDomain, YDomainVar,
      BaseDistr, W1, BaseProcess]]) =
    new MixtureMachine[T, I, Y, YDomain, YDomainVar, BaseDistr, W1, BaseProcess](model) {
      override val confToModel = confModelPipe
      override val mixturePipe = mixtPipe
    }
  
}
