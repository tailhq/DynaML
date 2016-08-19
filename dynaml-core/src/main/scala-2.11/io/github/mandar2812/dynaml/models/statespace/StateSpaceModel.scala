package io.github.mandar2812.dynaml.models.statespace

import java.io.Serializable
import POMP._
import breeze.linalg.DenseVector
import breeze.stats.distributions.Rand

trait StateSpaceModel extends Serializable {
  // The observation model
  def observation: Eta => Rand[Observation]
  // the link function
  def link(x: Gamma): Eta = Vector(x)
  // deterministic transformation, such as seasonality
  def f(s: State, t: Time): Gamma
  // initialise the SDE state
  def x0: Rand[State]
  // Step the SDE
  def stepFunction: (State, TimeIncrement) => Rand[State]
  // calculate the likelihood of the observation given the state
  def dataLikelihood: (Eta, Observation) => LogLikelihood
}

object StateSpaceModel {
  def op(mod1: Parameters => StateSpaceModel,
         mod2: Parameters => StateSpaceModel): Parameters => StateSpaceModel =
    p => new StateSpaceModel {

      def observation = x => p match {
        case BranchParameter(lp,_) => mod1(lp).observation(x)
        case param: LeafParameter => mod1(param).observation(x)
      }

      override def link(x: Double) = mod1(p).link(x)

      def f(s: State, t: Time) = s match {
        case BranchState(ls, rs) =>
          mod1(p).f(ls, t) + mod2(p).f(rs, t)
        case x: LeafState =>
          mod1(p).f(x, t)
      }

      def x0 = p match {
        case BranchParameter(lp, rp) =>
          for {
            l <- mod1(lp).x0
            r <- mod2(rp).x0
          } yield l |+| r
        case param: LeafParameter =>
          for {
            l <- mod1(param).x0
            r <- mod2(param).x0
          } yield l |+| r
      }

      def stepFunction = (s, dt) => (s, p) match {
        case (BranchState(ls, rs), BranchParameter(lp, rp)) =>
          for {
            l <- mod1(lp).stepFunction(ls, dt)
            r <- mod2(rp).stepFunction(rs, dt)
          } yield BranchState(l, r)
        case (x: LeafState, param: LeafParameter) => // Null model case, non-null must be on left
          mod1(param).stepFunction(x, dt)
      }

      def dataLikelihood = (s, y) => p match {
        case param: LeafParameter => mod1(param).dataLikelihood(s, y)
        case BranchParameter(lp, _) => mod1(lp).dataLikelihood(s, y)
      }
    }

  def zeroModel(stepFun: SdeParameter => (State, TimeIncrement) => Rand[State]): Parameters => StateSpaceModel = p => new StateSpaceModel {
    def observation = x => new Rand[Observation] { def draw = x.head }
    def f(s: State, t: Time) = s.head
    def x0 = new Rand[State] { def draw = LeafState(DenseVector[Double]()) }
    def stepFunction = p match {
      case LeafParameter(_,_,sdeparam  @unchecked) => stepFun(sdeparam)
    }

    def dataLikelihood = (s, y) => 0.0
  }
}
