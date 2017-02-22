package io.github.mandar2812.dynaml.probability

import org.apache.commons.math3.linear.{Array2DRowRealMatrix, CholeskyDecomposition, LUDecomposition}
import shapeless.Witness
import spire.algebra._
import spire.random.{Dist, Gaussian, Generator, Uniform}
import spire.std.seq._
import spire.syntax.innerProductSpace._
import spire.syntax.order._

abstract class HMC[R : Uniform : Gaussian, N, G, D <: Int with Singleton : Witness.Aux](val posterior: Tree[R, N] => (R, G), val M: Matrix[D, R], val alpha: R, val eps: R, val L: Int, val RToDouble: R => Double)(implicit val rng: Generator, implicit val f: Field[R], implicit val trig: Trig[R], implicit val n: NRoot[R], implicit val s: Signed[R], implicit val o: Order[R]) extends (Z[R, N, G] => Z[R, N, G]) {

  type ZZ = Z[R, N, G]

  val (invM, choleskyL): (Matrix[D, R], Matrix[D, R]) = {
    val apacheM = new Array2DRowRealMatrix(M.size, M.size)
    M.indices.foreach(Function.tupled((i, j) => apacheM.setEntry(i, j, RToDouble(M(i, j)))))
    val apacheInvM = new LUDecomposition(apacheM).getSolver.getInverse
    val apacheCholeskyL = new CholeskyDecomposition(apacheM).getL
    (Matrix[D, R]((i: Int, j: Int) => Field[R].fromDouble(apacheInvM.getEntry(i, j))), Matrix[D, R]((i: Int, j: Int) => Field[R].fromDouble(apacheCholeskyL.getEntry(i, j))))
  }

  val uniform = Dist.uniform(Field[R].zero, Field[R].one)
  val gaussian = Dist.gaussian(Field[R].zero, Field[R].one)
  val sqrtalpha = NRoot[R].sqrt(alpha)
  val sqrt1malpha = NRoot[R].sqrt(1 - alpha)

  def U(q: Tree[R, N]): (R, G)

  def K(p: IndexedSeq[R]): (R, G)

  def flipMomentum(z: ZZ): ZZ = {
    val pp = -z.p
    z.copy(p = pp)(_K = K(pp))
  }

  def corruptMomentum(z: ZZ): ZZ = {
    val r = IndexedSeq.fill(z.p.size)(rng.next(gaussian))
    val pp = sqrt1malpha *: z.p + sqrtalpha *: (choleskyL * r)
    z.copy(p = pp)(_K = K(pp))
  }

  def simulateDynamics(z: ZZ): ZZ

  override def apply(z: ZZ): ZZ = {
    val zp = flipMomentum(simulateDynamics(z))
    val a = Trig[R].exp(z.H - zp.H) min 1
    corruptMomentum(flipMomentum(if (rng.next(uniform) < a) zp else z))
  }

}
