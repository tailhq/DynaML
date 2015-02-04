package org.kuleuven.esat

/**
 * Created by mandar on 4/2/15.
 */
package object utils {
  val log1pExp: (Double) => Double = (x) => {x + math.log1p(math.exp(-x))}
}
