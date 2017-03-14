package io.github.mandar2812.dynaml.probability

import spire.std.seq._
import spire.syntax.vectorSpace._

/**
  * LeapFrog method
  *
  * @tparam R Represents an orthant of dimension one
  * @tparam N Number of leaves on evolutionary tree
  * @tparam D Leaf set in evolutionary tree
  *
  */
  
trait LeapFrog[R, N, D <: Int with Singleton] extends NumericalDynamics[R, N, D] {

  def leapprog(eps: R)(z: ZZ): ZZ = {
    val halfEps = eps / 2 // Initialize a half step 
    val pp = z.p - halfEps *: z.dU // Calculates momentum as it take a half step in time to update the momentum variable
    val (_, dK) = K(pp) // simulates postion variable
    val qp = z.q.modifyLengths(_ + eps *: dK) // Takes full step in time to update the position variable
    val Up = U(qp)
    val ppp = pp - halfEps *: Up._2 // Calculate the remaining half step in time to finish updating the momentum variable
    Z(qp, ppp)(Up, K(ppp))
  }

}
