package model

import model.POMP._
import breeze.linalg.DenseVector

sealed trait State {
  import State._

  def map(f: DenseVector[Double] => DenseVector[Double]): State = State.map(this)(f)
  def flatten: Vector[Double] = State.flatten(this)
  def head: Double = State.head(this)
  def |+|(that: State): State = {
    combine(this, that)
  }
  def isEmpty: Boolean = State.isEmpty(this)
  override def toString: String = this.flatten.mkString(", ")
}
case class LeafState(data: DenseVector[Double]) extends State with Serializable
case class BranchState(left: State, right: State) extends State with Serializable

object LeafState {
  def apply(a: Double): LeafState = {
    new LeafState(DenseVector(a))
  }
}

object State {
  def combine(state1: State, state2: State): State = {
    if (state1.isEmpty) {
      state2
    } else if (state2.isEmpty) {
      state1
    } else {
      BranchState(state1, state2)
    }
  }

  def zero: State = {
    LeafState(DenseVector[Double]())
  }

  def head(s: State): Double = s match {
    case LeafState(x) => x(0)
    case BranchState(l, _) => head(l)
  }

  /**
    * Determines if a state contains 
    */
  def isEmpty(state: State): Boolean = state match {
    case LeafState(x) => x.length == 0
    case BranchState(lp, rp) => isEmpty(lp) && isEmpty(rp)
  }

  def toList(s: State): List[DenseVector[Double]] = s match {
    case BranchState(l, r) => toList(l) ::: toList(r)
    case LeafState(x) => List(x)
  }

  /**
    * Get the node element at position n from the left, indexed from 0
    * @param n the node position from the left
    */
  def getState(s: State, n: Int): LeafState = {
    val l = toList(s)
    LeafState(l(n))
  }

  /**
    * Maps all the values contained inside of all leaf nodes
    * @param s a given tree of states
    * @param f a function from a vector to a vector, usually defined using map, eg. x => x map (_ + 1)
    * @return the state in the same structure only changed by the provided f
    */
  def map(s: State)(f: DenseVector[Double] => DenseVector[Double]): State = s match {
    case LeafState(x) => LeafState(f(x))
    case BranchState(l, r) => BranchState(map(l)(f), map(r)(f))
  }

  /**
    * Calculate the weighted mean of a list of States
    * @param x a list of States
    * @param w their associated weights
    * @return the weighted mean
    */
  def weightedMean(x: Vector[State], w: Vector[Double]): State = {

    val normalisedWeights = w map (_ / w.sum)
    val st = x.zip(normalisedWeights) map {
      case (s, weight) =>
        def loop(s: State, w: Double): State =
          s match {
            case LeafState(state) => LeafState(state map (_ * w))
            case BranchState(ls, rs) => BranchState(loop(ls, w), loop(rs, w))
          }
        loop(s, weight)
    }
    st.reduceLeft((a: State, b: State) => addStates(a,b))
  }

  /**
    * Add two states with the same structure, used in weighted mean
    */
  def addStates(
    s1: State,
    s2: State): State =

    (s1, s2) match {
      case (x: LeafState, y: LeafState) if x.isEmpty => y
      case (x: LeafState, y: LeafState) if y.isEmpty => x
      case (LeafState(x), LeafState(x1)) => LeafState(x + x1)
      case (BranchState(l, r), BranchState(l1, r1)) => BranchState(addStates(l, l1), addStates(r, r1))
    }

  def flatten(s: State): Vector[Double] =
    s match {
      case LeafState(x) => x.data.toVector
      case BranchState(ls, rs) => flatten(ls) ++ flatten(rs)
    }
}
