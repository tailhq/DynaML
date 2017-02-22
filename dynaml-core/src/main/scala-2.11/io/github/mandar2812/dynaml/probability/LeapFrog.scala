package io.github.mandar2812.dynaml.probability

import spire.std.seq._
import spire.syntax.vectorSpace._

trait LeapFrog[R, N, D <: Int with Singleton] extends NumericalDynamics[R, N, D] {

  def leapprog(eps: R)(z: ZZ): ZZ = {
    val halfEps = eps / 2
    val pp = z.p - halfEps *: z.dU
    val (_, dK) = K(pp)
    val qp = z.q.modifyLengths(_ + eps *: dK)
    val Up = U(qp)
    val ppp = pp - halfEps *: Up._2
    Z(qp, ppp)(Up, K(ppp))
  }

}
