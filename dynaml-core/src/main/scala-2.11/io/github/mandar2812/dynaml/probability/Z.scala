package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import spire.syntax.field._

case class Z[R : Field, N, G](q: Tree[R, N], p: IndexedSeq[R])(_U: => (R, G), _K: => (R, G)) {
  lazy val (u, dU) = _U
  lazy val (k, dK) = _K
  lazy val H = u + k
  def copy(q: Tree[R, N] = this.q, p: IndexedSeq[R] = this.p)(_U: => (R, G) = (u, dU), _K: => (R, G) = (k, dK)): Z[R, N, G] = Z(q, p)(_U, _K)
}

object Z {

  def apply[R : Field, N, G](q: Tree[R, N], p: IndexedSeq[R], U: Tree[R, N] => (R, G), K: IndexedSeq[R] => (R, G)): Z[R, N, G] = Z(q, p)(U(q), K(p))

}
