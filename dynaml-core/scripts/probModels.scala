/**
  * Created by mandar on 28/7/16.
  */

import breeze.stats.distributions._
import spire.implicits._

val n = RandomVariable(Gaussian(0.0, 0.25))
val n1 = RandomVariable(Laplace(0.0, 0.25))

val combinedDist = n :*: n1

val cond2 = DataPipe((xy: (Double, Double)) => RandomVariable(new Gumbel(0.4*xy._1 + 0.6*xy._2, math.sin(xy._1+xy._2)/2.0)))

val mod = ProbabilityModel(combinedDist, cond2)

val combinedDist2 = n1 :*: n

val mod1 = ProbabilityModel(combinedDist2, cond2)


val cointoss = RandomVariable(new Binomial(1, 0.5))

val mixture = DataPipe((toss: Int) => if(toss == 1) mod else mod1)

val mixtureModel = ProbabilityModel(cointoss, mixture)
