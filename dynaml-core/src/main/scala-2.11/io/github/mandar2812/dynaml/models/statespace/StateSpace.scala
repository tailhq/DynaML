package io.github.mandar2812.dynaml.models.statespace

import POMP._
import breeze.stats.distributions.{Rand, Gaussian, MultivariateGaussian}
import breeze.linalg.{diag, DenseVector}
import breeze.numerics.{exp, sqrt}
import breeze.stats.distributions.{Gaussian, Rand}

object StateSpace {
    /**
    * Steps all the states using the identity
    * @param p a Parameter
    * @return a function from state, dt => State
    */
  def stepNull(p: Parameters): (State, TimeIncrement) => Rand[State] = {
    (s, dt) => new Rand[State] { def draw = State.map(s)(x => x) }
  }

  /**
    * A step function for generalised brownian motion, dx_t = mu dt + sigma dW_t
    * @param p an sde parameter
    * @return A function from state, time increment to state
    */
  def stepBrownian(p: SdeParameter): (State, TimeIncrement) => Rand[State] = {
    (s, dt) => p match {
      case BrownianParameter(mu, sigma) => {
        new Rand[State] {
          def draw =
            s map (x => DenseVector((x.data, mu.data, diag(sigma).toArray).zipped.
              map { case (a, m, sd) =>
                Gaussian(a + m * dt, Math.sqrt(sd * sd * dt)).draw
              }))
          }
      }
    }
  }

  /**
    * Steps the state by the value of the parameter "a" 
    * multiplied by the time increment "dt"
    * @param p a parameter Map
    * @return a function from (State, dt) => State, with the
    * states being the same structure before and after
    */
  def stepConstant(p: SdeParameter): (State, TimeIncrement) => Rand[State] = {
    (s, dt) => p match {
      case StepConstantParameter(a) =>
        new Rand[State] { def draw = s map (_ + (a :* dt)) }
    }
  }

  /**
    * A step function for the Ornstein Uhlenbeck process dx_t = - alpha x_t dt + sigma dW_t
    * @param p the parameters of the ornstein uhlenbeck process, theta, alpha and sigma
    * @return
    */

  def stepOrnstein(p: SdeParameter): (State, TimeIncrement) => Rand[State] = {
    (s, dt) =>  new Rand[State] {
      def draw = p match {
        case OrnsteinParameter(theta, alpha, sigma) =>
          s map { x => // calculate the mean of the solution
            val mean = (x.data, alpha.data, theta.data).zipped map { case (state, a, t) => t + (state - t) * exp(- a * dt) }
            // calculate the variance of the solution
            val variance = (sigma.data, alpha.data).zipped map { case (s, a) => (s*s/2*a)*(1-exp(-2*a*dt)) }
            DenseVector(mean.zip(variance) map { case (a, v) => Gaussian(a, sqrt(v)).draw() })
            }
      }
    }
  }
}
