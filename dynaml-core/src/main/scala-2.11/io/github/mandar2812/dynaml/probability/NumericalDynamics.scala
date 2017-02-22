package io.github.mandar2812.dynaml.probability

import spire.std.seq._
import spire.syntax.innerProductSpace._

trait NumericalDynamics[R, N, D <: Int with Singleton] extends HMC[R, N, IndexedSeq[R], D] {

  type G = IndexedSeq[R]

  def U(q: Tree[R, N]): (R, IndexedSeq[R]) = {
    val (p, dP) = posterior(q)
    (-p, -dP)
  }

  def K(p: IndexedSeq[R]): (R, IndexedSeq[R]) = {
    val invMp = invM * p
    ((p dot invMp) / 2, invMp)
  }

  def leapprog(eps: R)(z: Z[R, N, G]): Z[R, N, G]

  def simulateDynamics(z: Z[R, N, G]): Z[R, N, G] = (0 until L).foldLeft(z)((z, _) => leapprog(eps)(z))

}
