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

// Using properties of multivariate normal distribution
// The equations required to program up the Kalman filter
case class FilterOut(data: Data, p: Parameters)
 
def filter(d: Data, p: Parameters): FilterOut = {
// update the mean and variance of the posterior to determine the state space
val e1 = d.observation - p.m0
val a1 = (p.c0 + p.b)/(p.c0 + p.b + p.a)
val m1 = p.m0 + a1 * e1
val c1 = a1 * p.a
 
// return the data with the expectation of the hidden state and the updated Parameters
FilterOut(Data(d.time, d.observation, Some(m1)), Parameters(p.a, p.b, m1, c1))
}
def filterSeries(data: Seq[Data],initParams: Parameters): Seq[FilterOut] = {
 
val (a, b) = (initParams.a, initParams.b) // a and b are fixed
val initFilter = Vector[FilterOut](filter(data.head, initParams)) // initialise the filter
 
data.tail.foldLeft(initFilter)((acc, nextObservation) =&gt; {
// construct the parameters from the previous step of the filter
val p = Parameters(a, b, acc.head.p.m0, acc.head.p.c0)
 
// add the filtered observation to the head of the list
filter(nextObservation, p) +: acc
}).reverse
}
