package io.github.mandar2812.dynaml.wavelets

import spire.std.any._
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 15-7-16.
  */
class WaveletSpec extends FlatSpec with Matchers {
  "A Haar Wavelet" should "have consistent values" in {
    val motherWav = (x: Double) =>
      if (x >= 0.0 && x < 0.5) 1.0
      else if (x >= 0.5 && x < 1.0) -1.0
      else 0.0

    val hWavelet = new Wavelet[Double](motherWav)(1.0, 0.5)

    assert(hWavelet(1.0) == -1.0)
  }
}
