/*
* External Imports
* */
//Ammonite imports
import ammonite.ops._
//Import breeze for linear algebra
import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
//Apache Spark for big data support
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
//Load Wisp-Highcharts for plotting
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
//Import spire implicits for definition of
//fields, algebraic structures on primitive types
import spire.implicits._
/*
* DynaML imports
* */
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.algebra._
//The pipes API
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.DynaMLPipe._
//Load the DynaML model api members
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.models.svm._
import io.github.mandar2812.dynaml.models.lm._
//Utility functions
import io.github.mandar2812.dynaml.utils
//Kernels for GP,SVM models
import io.github.mandar2812.dynaml.kernels._
//Shell examples
import io.github.mandar2812.dynaml.examples._
//Load neural net primitives
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._
//The probability API
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.probability.distributions._
//Wavelet API
import io.github.mandar2812.dynaml.wavelets._
//OpenML support
import io.github.mandar2812.dynaml.openml.OpenML
//Spark support
import io.github.mandar2812.dynaml.DynaMLSpark._
//Tensorflow support
import io.github.mandar2812.dynaml.tensorflow._
//Renjin imports
import javax.script._
import org.renjin.script._
import org.renjin.sexp._
import utils.Renjin._