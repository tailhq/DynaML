// A 1st order Polynomial DLM with constant a and b
// scalaz - a funtional library for Scala
// Gaussian distribution from https://github.com/scalanlp/breeze/

import scalaz._
import Scalaz._
import breeze.stats.distributions.Gaussian
 
case class Data(time: Time, observation: Observation, state: Option[State])
case class Parameters(a: Double, b: Double, m0: Double, c0: Double)
 
def simulate(p: Parameters): Stream[Data] = {
val stateSpace = unfold(Gaussian(p.m0, sqrt(p.c0)).draw)(x =&gt; 
Some(x, x + Gaussian(0, sqrt(p.w)).draw)
)
stateSpace.zipWithIndex map { case (x, t) =&gt;
Data(t, x + Gaussian(0, sqrt(p.a)).draw, Some(x)) }
}
 
val p = Parameters(3.0, 0.5, 0.0, 10.0)
// simulate 16 different realisations of 100 observations, representing 16 stations
val data = (1 to 16) map (id =&gt; (id, simulate(p).take(100).toVector))
