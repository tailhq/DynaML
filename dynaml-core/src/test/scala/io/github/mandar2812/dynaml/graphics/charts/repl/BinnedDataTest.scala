package io.github.mandar2812.dynaml.graphics.charts.repl

import org.scalatest.Matchers
import org.scalatest.FunSuite
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._

/**
 * User: austin
 * Date: 1/30/15
 */
class BinnedDataTest extends FunSuite with Matchers  {

  test("Binned list should have one item per bucket") {
    val bd: BinnedData = binIterableNumBins(List(1, 2, 3, 4), 4)
    bd.toBinned().toArray should be (Array(("1.00 - 2.00", 1), ("2.00 - 3.00", 1), ("3.00 - 4.00", 1), ("4.00 - 5.00", 1)))
  }

  test("Empty buckets should receive a count of zero") {
    val bd: BinnedData = binIterableNumBins(List(1, 2, 4), 4)
    bd.toBinned().toArray should be (Array(("1.00 - 2.00", 1), ("2.00 - 3.00", 1), ("3.00 - 4.00", 0), ("4.00 - 5.00", 1)))
  }

  test("Triplet should be binned") {
    val bd: BinnedData = mkTrueTriplet(List((1, 2, 4), (2, 3, 5)))
    bd.toBinned().toArray should be (Array(("1 - 2", 4), ("2 - 3", 5)))
  }

  test("SAT scores should create the proper x-axis categories") {
    val hc = histogram(List(490, 499, 459, 575, 575, 513, 382, 525, 510, 542, 368, 564, 509, 530, 485, 521, 495, 526, 474, 500, 441,
      750, 495, 476, 456, 440, 547, 479, 501, 476, 457, 444, 444, 467, 482, 449, 464, 501, 670, 740, 590, 700, 590, 450, 452,
      468, 472, 447, 520))

    hc.xAxis.get.head.categories.get should be (Array("368.00 - 431.67","431.67 - 495.33","495.33 - 559.00","559.00 - 622.67","622.67 - 686.33","686.33 - 750.00","750.00 - 813.67"))
  }
}
