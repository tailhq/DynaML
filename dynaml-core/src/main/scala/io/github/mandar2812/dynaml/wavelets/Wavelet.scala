package io.github.mandar2812.dynaml.wavelets

import spire.algebra.{Field, NRoot}

/**
  * Created by mandar on 14-7-16.
  */
class Wavelet[I](mother: (I) => I)(scale: I, shift: I)(implicit ev: Field[I], ev1: NRoot[I]) {

  val psi = mother

  val (lambda,a): (I,I) = (shift,scale)

  def apply(x: I): I = Wavelet(psi)(a, lambda).apply(x)

}

object Wavelet {

  def apply[I](mother: (I) => I)(scale: I, shift: I)(implicit ev: Field[I], ev1: NRoot[I]): (I) => I =
    (x: I) => ev.div(mother(ev.div(ev.minus(x, shift), scale)), ev1.sqrt(scale))
}


