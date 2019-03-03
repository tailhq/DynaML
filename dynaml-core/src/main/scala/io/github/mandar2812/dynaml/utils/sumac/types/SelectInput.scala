package io.github.mandar2812.dynaml.utils.sumac.types

import collection.mutable.LinkedHashSet

class SelectInput[T](var value: Option[T], val options: LinkedHashSet[T])

object SelectInput{
  def apply[T](options: T*) = new SelectInput[T](value = None, options = (LinkedHashSet.empty ++ options))
  def apply[T](value: Option[T], options: Traversable[T]) = new SelectInput[T](value = value, options = (LinkedHashSet.empty ++ options))
}
