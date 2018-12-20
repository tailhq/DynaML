package io.github.mandar2812.dynaml.models.lm

import java.io.{File, PrintWriter}

import breeze.stats.distributions.Gaussian
import io.github.mandar2812.dynaml.models.lm.generalDLM._

import scala.io.Source

object KFilter {

  case class FilterOut(data: Data, p: Parameters, likelihood: Loglikelihood) {
    override def toString = data.toString + ", " + p.toString
  }

  def filterll(data: Seq[Data])(params: Parameters): Loglikelihood = {
    val (a, b) = (params.a, params.b) // a and b are fixed
    val initFilter = Vector[FilterOut](filter(data.head, params)) // initialise the filter

    val filtered = data.tail.foldLeft(initFilter)((acc, nextObservation) => {
      // construct the parameters from the previous step of the filter
      val p = Parameters(a, b, acc.head.p.l1, acc.head.p.l2)

      // add the filtered observation to the head of the list
      filter(nextObservation, p) +: acc
    }).reverse

    // sum the values of the likelihood
    filtered.map(_.likelihood).sum
  }

  def filter(d: Data, p: Parameters): FilterOut = {
    // update the mean and variance of the posterior to determine the state space
    val e1 = d.observation - p.l1
    val a1 = (p.l2 + p.b)/(p.l2 + p.b + p.a)
    val m1 = p.l1 + a1 * e1
    val c1 = a1 * p.a

    val likelihood = Gaussian(p.l1, p.l2 + p.b + p.a).logPdf(d.observation)

    // return the data with the expectation of the hidden state and the updated Parameters
    FilterOut(Data(d.time, d.observation, Some(m1)), Parameters(p.a, p.b, m1, c1), likelihood)
  }

  def filterSeries(data: Seq[Data])(params: Parameters): Seq[FilterOut] = {

    val (a, b) = (params.a, params.b) // a and b are fixed
    val initFilter = Vector[FilterOut](filter(data.head, params)) // initialise the filter

    data.tail.foldLeft(initFilter)((acc, nextObservation) => {
      // construct the parameters from the previous step of the filter
      val p = Parameters(a, b, acc.head.p.l1, acc.head.p.l2)

      // add the filtered observation to the head of the list
      filter(nextObservation, p) +: acc
    }).reverse
  }

  val runKFilter = {
    val p = Parameters(3.0, 0.5, 0.0, 10.0)
    // simulate 16 different realisations of 100 observations, representing 16 stations
    val observations = (1 to 16) map (id => id -> simulate(p).take(100).toVector)

    // filter for one station, using simulated data
    observations.
      filter{ case (id, _) => id == 1 }.
      flatMap{ case (_, d) => filterSeries(d)(p) }

    // or, read in data from the file we previously wrote
    val data = Source.fromFile("data/firstOrderdlmRes.csv").getLines.toList.
      map(l => l.split(",")).
      map(r => r(0).toInt -> Data(r(1).toDouble, r(2).toDouble, None))

    // filter for all stations, using data from file
    val filtered = data.
      groupBy{ case (id, _) => id }. //groups by id
      map{ case (id, idAndData) =>
        (id, idAndData map (x => x._2)) }. // changes into (id, data) pairs
      map{ case (id, data) =>
        (id, filterSeries(data.sortBy(_.time))(p)) } // apply the filter to the sorted data

    // write the filter for all stations to a file
    val pw = new PrintWriter(new File("data/filteredDlmRes.csv"))
    pw.write(filtered.
      flatMap{ case (id, data) =>
          data map (x => id + ", " + x.toString)}.
      mkString("\n"))
    pw.close()
  }
}
