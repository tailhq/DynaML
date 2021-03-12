package io.github.tailhq.dynaml.utils.sumac.types

import scala.collection._

class MultiSelectInput[T](var value: Set[T], val options: mutable.LinkedHashSet[T])

object MultiSelectInput {
  def apply[T](options: T*) = new MultiSelectInput[T](Set(), options = (mutable.LinkedHashSet.empty ++ options))
}


