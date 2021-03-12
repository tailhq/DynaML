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
package io.github.tailhq.dynaml.optimization

import breeze.linalg._
import breeze.numerics.lgamma
import io.github.tailhq.dynaml.{DynaMLPipe, utils}
import io.github.tailhq.dynaml.algebra.square
import io.github.tailhq.dynaml.analysis.VectorField
import io.github.tailhq.dynaml.pipes.{DataPipe, Encoder}
import io.github.tailhq.dynaml.probability.{GaussianRV, MultGaussianRV}

/**
  * <h4>Covariance Matrix Adaptation - Evolutionary Search</h4>
  *
  * An implementation of the (&mu;/&mu;<sub>W</sub>, &lambda;) CMA-ES
  * optimization algorithm as described in
  * <a href="https://arxiv.org/pdf/1604.00772.pdf">https://arxiv.org/pdf/1604.00772.pdf</a>
  *
  * @param model The model having hyper-parameters. Must extend [[GloballyOptimizable]]
  * @param hyper_parameters A list of model hyper-parameters.
  * @param learning_rate Usually set to a value of 1, or some value less than 1.
  * @param hyp_parameter_scaling In case the search space is not the whole Euclidean
  *                              space, set a one-to-one mapping from the hyper-parameter
  *                              space to a latent Euclidean space.
  * */
class CMAES[T <: GloballyOptimizable](
  model: T,
  val hyper_parameters: Seq[String],
  val learning_rate: Double = 1d,
  hyp_parameter_scaling: Option[Map[String, Encoder[Double, Double]]] = None)
  extends ModelTuner[T, T] {


  protected var MAX_ITERATIONS: Int = 10

  val problem_size = hyper_parameters.length

  val config = CMAES.get_default_config(problem_size)

  /**
    * Set the maximum number of generations to run.
    * */
  def setMaxIterations(m: Int): this.type = {
    MAX_ITERATIONS = m
    this
  }

  override val system: T = model

  private val hyp_map_to_vec: Encoder[CMAES.HyperParams, DenseVector[Double]] = CMAES.map_to_vec_enc(hyper_parameters)

  private val process_hyp: Encoder[CMAES.HyperParams, CMAES.HyperParams] = hyp_parameter_scaling match {
    case None => Encoder(
      DynaMLPipe.identityPipe[CMAES.HyperParams],
      DynaMLPipe.identityPipe[CMAES.HyperParams])

    case Some(scaling) => Encoder(
      DataPipe[CMAES.HyperParams, CMAES.HyperParams](config => config.map(kv => (kv._1, scaling(kv._1)(kv._2)))),
      DataPipe[CMAES.HyperParams, CMAES.HyperParams](config => config.map(kv => (kv._1, scaling(kv._1).i(kv._2))))
    )
  }

  private val hyp_encoder = process_hyp > hyp_map_to_vec

  /**
    * Evaluate the fitness of a population sample.
    * @param sample A population sample
    * @return A sorted sequence of fitness values
    *         and candidates.
    *
    * */
  def evaluate_population(sample: Seq[DenseVector[Double]]): Seq[(Double, DenseVector[Double])] =
    sample.map(s => {

      val config = hyp_map_to_vec.i(s)

      println("Evaluating configuration: ")
      pprint.pprintln(hyp_encoder.i(s))

      val e = system.energy(config)

      print("Energy = ")
      pprint.pprintln(e)

      (e, s)
    }).sortBy(_._1)


  /**
    * Run [[MAX_ITERATIONS]] generations of CMA-ES.
    *
    * @param initialConfig Used as the m<sub>0</sub>, the
    *                      initial sampling mean.
    * @param options Extra options for the algorithm,
    *                set the variance scaling as
    *                {{{
    *                  Map("stepSize" -> 1.5)
    *                }}}
    * */
  override def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String]): (T, Map[String, Double]) = {

    val sigma = if(options.isDefinedAt("stepSize")) options("stepSize").toDouble else 1.0

    val initialState = CMAES.State(
      hyp_encoder(initialConfig),
      DenseMatrix.eye[Double](problem_size),
      DenseVector.zeros[Double](problem_size), sigma,
      DenseVector.zeros[Double](problem_size))

    println("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    println("Covariance Matrix Adaptation - Evolutionary Search (CMA-ES): ")
    print("Population Size: ")
    pprint.pprintln(config.λ)
    print("Parent Number: ")
    pprint.pprintln(config.µ)
    println("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    println()

    println("Configuration: ")
    pprint.pprintln(config)

    val result = (0 until MAX_ITERATIONS).foldLeft((initialState, Seq[DenseVector[Double]]()))(
      (generation: (CMAES.State, Seq[DenseVector[Double]]), it: Int) => {

        println("------------------------------------------------------------------------")
        print("Generation ")
        pprint.pprintln(it)

        val (state, _) = generation

        pprint.pprintln(state.C)

        val population = CMAES.sample_population(state, config)

        val ranked_population = evaluate_population(population)

        val new_state = CMAES.update_state(learning_rate, config)(state, ranked_population.map(_._2), it)

        println("------------------------------------------------------------------------")
        println("New State: ")
        pprint.pprintln(new_state)
        println("------------------------------------------------------------------------")

        (new_state, ranked_population.map(_._2))
      })

    println("Optimum configuration: ")
    pprint.pprintln(hyp_encoder.i(result._2.head))

    (system, hyp_encoder.i(result._2.head))
  }
}


object CMAES {

  type HyperParams = Map[String, Double]

  def map_to_vec_enc(h: Seq[String]): Encoder[HyperParams, DenseVector[Double]] = Encoder(
    (config: HyperParams) => DenseVector(h.map(config).toArray),
    (vec: DenseVector[Double]) => h.zip(vec.toArray).toMap
  )

  private[optimization] case class State(
    m: DenseVector[Double], C: DenseMatrix[Double],
    p_c: DenseVector[Double], sigma: Double,
    p_s: DenseVector[Double]
  )

  /**
    * Represents a CMA-ES configuration.
    *
    * @param n The problem size.
    * @param λ The population size.
    * @param µ The number of parents to choose.
    * @param µ_eff The <i>variance effective selection mass</i>,
    *              should be set to
    *              (&Sigma;<sub>i</sub>|w<sub>i</sub>|)<sup>2</sup>/(&Sigma;w<sub>i</sub><sup>2</sup>)
    *
    * @param w The weights assigned to each ranked candidate.
    *          Assumed to be in the order of fitness value.
    *
    * @param c_s Exponential smoothing factor for the step size.
    * @param d_s Scaling constant in exponential smoothing of step size.
    * @param c_c Smoothing parameter for updating the evolution path [[State.p_c]]
    * @param c1 Parameter used in update of covariance matrix.
    * @param c_µ Multiplicative factor used to scale sum of weights.
    *
    * */
  case class Config(
    n: Int, λ: Int, µ: Int, µ_eff: Double, w: DenseVector[Double],
    c_s: Double, d_s: Double, c_c: Double, c1: Double,
    c_µ: Double
  )


  def update_state(
    learning_rate: Double,
    config: Config)(
    state: State,
    population: Seq[DenseVector[Double]],
    g: Int): State = {

    val y = population.map(_ - state.m)

    val y_w = config.w(0 until config.µ).toArray.zip(y.slice(0, config.µ)).map(c => c._2*c._1).reduce(_ + _)

    //Update the mean of the sampling distribution
    val m = state.m + y_w*state.sigma*learning_rate

    val L = cholesky(state.C)

    val Exp_N01 = math.sqrt(2d)*math.exp(lgamma((config.n + 1)/2d) - lgamma(config.n/2d))

    //Update the step size control parameters
    val p_s = (1 - config.c_s)*state.p_s + (L\y_w)*math.sqrt(config.c_s*(2 - config.c_s)*config.µ_eff)
    val sigma = state.sigma*math.exp(config.c_s*(norm(p_s, 1)/Exp_N01 - 1)/config.d_s)

    //Compute primitives needed for covariance adaptation.
    val h_s =
      if(norm(p_s, 1)/math.sqrt(1d - math.pow(1 - config.c_s, 2*(g+1))) < (1.4 + 2/(config.n + 1))*Exp_N01) 1.0
      else 0d

    val p_c = state.p_c*(1d - config.c_c) + y_w*(h_s*math.sqrt(config.c_s*(2 - config.c_s)*config.µ_eff))

    //Adjust the weights.
    val w_adj = config.w.toArray.zip(y).map(c => if(c._1 > 0d) c._1 else c._1 * config.n/math.pow(norm(L\c._2), 2d))

    def delta(x: Double): Double = (1d - x)*config.c_c*(2 - config.c_c)

    //Update the covariance matrix
    val C =
      state.C*(1d + config.c1*delta(h_s) - config.c1 - config.c_µ*sum(config.w)) +
        (p_c*p_c.t)*config.c1 +
        w_adj.map(x => x*config.c_µ).zip(y).map(c => (c._2*c._2.t)*c._1).reduce(_ + _)


    State(m, C, p_c, sigma, p_s)
  }

  /**
    * Generate a population sample.
    * @param state The algorithm state as a [[State]] instance.
    * @param config The configuration of the algorithm.
    * @return A population sample.
    * */
  def sample_population(state: State, config: Config): Seq[DenseVector[Double]] = {

    implicit val f = VectorField(config.n)

    val pop_random_variable = MultGaussianRV(state.m, math.pow(state.sigma, 2d)*state.C)

    pop_random_variable.iid(config.λ).draw
  }

  /**
    * Generate the default configuration [[Config]] of CMA-ES.
    * @param n The problem size, i.e. the number of
    *          hyper-parameters to optimize.
    * */
  def get_default_config(n: Int): Config = {

    val λ = 4 + math.floor(3*math.log(n)).toInt


    val µ = math.floor(λ/2.0).toInt

    val w1 = DenseVector.tabulate[Double](λ)(i => math.log((λ+1d)/2) - math.log(i+1))

    val µ_eff: Double = math.pow(sum(w1), 2d)/square(sum(w1))

    val w1_pos = w1(0 until µ+1)
    val w1_pos_sum = sum(w1_pos)


    val w1_neg = w1(µ+1 to -1)
    val w1_neg_sum = -sum(w1_pos)

    val µ_eff_neg: Double = math.pow(sum(w1_neg), 2d)/square(sum(w1_neg))

    val alpha_cov = 2d

    val c_c = (4d + µ_eff/n)/(n + 4d + 2*µ_eff/n)

    val c1 = alpha_cov/(µ_eff + math.pow(n + 1.3, 2d))

    val c_µ = math.min(1d - c1, alpha_cov*(µ_eff - 2d + (1/µ_eff))/(math.pow(n+2, 2d) + alpha_cov*µ_eff/2d))


    val c_s = (µ_eff + 2d)/(n + µ_eff + 5)

    val d_s = 1 + 2*math.max(0d, math.sqrt((µ_eff - 1d)/(n + 1d)) - 1d) + c_s

    val alpha_µ_neg = 1d + c1/c_µ

    val alpha_µ_eff_neg = 1d + 2*µ_eff_neg/(µ_eff + 2d)

    val alpha_neg_posdef = (1d - c1 - c_µ)/(n*c_µ)


    val w = w1.map(x =>
      if(x >= 0d) x/w1_pos_sum
      else x*Seq(alpha_µ_neg, alpha_µ_eff_neg, alpha_neg_posdef).min/w1_neg_sum)

    Config(n, λ, µ, µ_eff, w, c_s, d_s, c_c, c1, c_µ)
  }

}