package io.github.mandar2812.dynaml.kernels

import breeze.generic.UFunc
import breeze.math.Semiring

/**
  * @author mandar2812 date:26/10/2016.
  *
  * Marker for some kernel operation.
  */
sealed trait KernelOpType
sealed trait KernelOpAdd extends KernelOpType
sealed trait KernelOpMult extends KernelOpType
sealed trait KernelOuterMult extends KernelOpType
sealed trait KernelOuterAdd extends KernelOpType

object KernelOpAdd extends KernelOpAdd with UFunc {
  implicit def opAddFromSemiring[S:Semiring]: Impl2[S, S, S] = new Impl2[S, S, S] {
    def apply(v: S, v2: S): S = implicitly[Semiring[S]].+(v, v2)
  }
}

object KernelOpMult extends KernelOpMult with UFunc {
  implicit def opMultFromSemiring[S:Semiring]: Impl2[S, S, S] = new Impl2[S, S, S] {
    def apply(v: S, v2: S): S = implicitly[Semiring[S]].*(v, v2)
  }
}

object KernelOuterMult extends KernelOuterMult with UFunc

object KernelOuterAdd extends KernelOuterAdd with UFunc