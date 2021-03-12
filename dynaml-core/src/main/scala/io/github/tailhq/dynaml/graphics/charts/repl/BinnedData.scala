package io.github.tailhq.dynaml.graphics.charts.repl

/**
 * User: austin
 * Date: 1/29/15
 *
 * Uses the magnet pattern to bin data, in preparation for building a histogram
 */
trait BinnedData {
  def toBinned(): Iterable[(String, Double)]

  def coupledTripletBinned[A, B, C](data: Iterable[((A, B), C)])(implicit ev: Numeric[C]) = {
    data.map{case((a, b), c) => s"$a - $b" -> ev.toDouble(c)}
  }
}

class PairBinned[A, B: Numeric](data: Iterable[(A, B)])(implicit ev: Numeric[B]) extends BinnedData {
  def toBinned(): Iterable[(String, Double)] = data.map{case(a, b) => a.toString -> ev.toDouble(b)}
}

class TrueTripletBinned[A, B, C: Numeric](data: Iterable[(A, B, C)]) extends BinnedData {
  def toBinned(): Iterable[(String, Double)] = coupledTripletBinned(data.map{case(a, b, c) => ((a, b), c)})
}

class CoupledTripletBinned[A, B, C: Numeric](data: Iterable[((A, B), C)]) extends BinnedData {
  def toBinned(): Iterable[(String, Double)] = coupledTripletBinned(data)
}

class IterableBinned[A: Numeric](data: Iterable[A], numBins: Int = -1) extends BinnedData {

  // Conditionally format on number of leading zeros
  // To support bins like 0.0001875 - 0.0002
  // While preserving 1.0 - 2.0 instead of 1.000 - 2.000
  def leadingZeros(x: Double) = {
    math.max(2, 1 + (math.log10(x) * -1)).toInt
  }

  def format(start: Double, end: Double, digits: Int) = {
    digits match {
      case x if x<=2 => f"$start%.2f - $end%.2f"
      case 3 => f"$start%.3f - $end%.3f"
      case 4 => f"$start%.4f - $end%.4f"
      case 5 => f"$start%.5f - $end%.5f"
      case _ => f"$start%.6f - $end%.6f"
    }
  }

  def toBinned(): Iterable[(String, Double)] = {
    def numericToDouble[X](x: X)(implicit ev: Numeric[X]): Double = ev.toDouble(x)

    val doubleData = data.map(numericToDouble(_)).toSeq.sorted

    val (min, max) = (doubleData.min, doubleData.max)

    def minDouble(x: Double, y: Double): Double = if(x<y) x else y
    val margin = doubleData.dropRight(1).zip(doubleData.drop(1)).map{case(left, right) => right - left}.reduce(minDouble) / 2

    val binWidth = ((max+margin) - (min-margin)) / numBins
    val zeros = leadingZeros(binWidth)

    val binMap = (min to max by binWidth).zipWithIndex.map{case(bin, index) => index -> bin}.toMap

    // This strategy risks short-changing the last bin - perhaps there is a more fair way to do it?
    def toBin(d: Double) = {
      val index = ((d - min) / binWidth).toInt
      binMap(index)
    }

    val binCounts = doubleData.map(toBin).groupBy(identity).map{case(bin, group) =>
      bin -> group.size.toDouble
    }

    binMap.toSeq.sortBy(_._1).map{case(index, bin) =>
      format(bin, bin+binWidth, zeros) -> binCounts.getOrElse(bin, 0d)
    }
  }
}

trait BinnedDataLowerPriorityImplicits {
  implicit def binIterable[A: Numeric](data: Iterable[A]): BinnedData = {
    val numElements = data.toSeq.distinct.size
    val numBins = math.sqrt(numElements).toInt
    new IterableBinned[A](data, numBins)
  }
}

object BinnedData {
  implicit def binIterableNumBins[A: Numeric](data: Iterable[A], numBins: Int): BinnedData = new IterableBinned[A](data, numBins)
  implicit def mkPair[A, B: Numeric](data: Iterable[(A, B)]) = new PairBinned(data)
  implicit def mkTrueTriplet[A, B, C: Numeric](data: Iterable[(A, B, C)]) = new TrueTripletBinned(data)
  implicit def mkCoupledTriplet[A, B, C: Numeric](data: Iterable[((A, B), C)]) = new CoupledTripletBinned(data)

  implicit def binIterableNumBins[A: Numeric](data: Array[A], numBins: Int): BinnedData = new IterableBinned[A](data.toSeq, numBins)
  implicit def mkPair[A, B: Numeric](data: Array[(A, B)]) = new PairBinned(data.toSeq)
  implicit def mkTrueTriplet[A, B, C: Numeric](data: Array[(A, B, C)]) = new TrueTripletBinned(data.toSeq)
  implicit def mkCoupledTriplet[A, B, C: Numeric](data: Array[((A, B), C)]) = new CoupledTripletBinned(data.toSeq)
}