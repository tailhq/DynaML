package io.github.mandar2812.dynaml

import scalaxy.streams.optimize
import spire.algebra.InnerProductSpace
import scala.collection.Bag

/**
  * Created by mandar on 11/01/2017.
  */
package object probability {

  var candidates: Int = 10000
 
  type Pattern = Map[Taxon, Int]
  val Pattern = Map
  type Patterns = Bag[Pattern]
  val Patterns = Bag

  def E[@specialized(Double) I](rv: RandomVariable[I])(implicit f: InnerProductSpace[I, Double]): I = optimize {
    f.divr(
      rv.iid(candidates)
        .sample()
        .reduce(
          (x, y) => f.plus(x,y)
        ),
      candidates.toDouble)
  }


  def E(rv: RandomVariable[Double]): Double = optimize {
    rv.iid(candidates).sample().sum/candidates.toDouble
  }


}
