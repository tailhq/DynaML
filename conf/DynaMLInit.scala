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
import io.github.tailhq.dynaml.graphics.charts.Highcharts._
//Import spire implicits for definition of
//fields, algebraic structures on primitive types
import spire.implicits._
/*
* DynaML imports
* */
import io.github.tailhq.dynaml.analysis.VectorField
import io.github.tailhq.dynaml.analysis.implicits._
import io.github.tailhq.dynaml.algebra._
//Load 3d graphics capabilities
import io.github.tailhq.dynaml.graphics.plot3d
//The pipes API
import io.github.tailhq.dynaml.pipes._
import io.github.tailhq.dynaml.DynaMLPipe
import io.github.tailhq.dynaml.DynaMLPipe._
//Load the DynaML model api members
import io.github.tailhq.dynaml.models._
import io.github.tailhq.dynaml.models.neuralnets._
import io.github.tailhq.dynaml.models.svm._
import io.github.tailhq.dynaml.models.lm._
//Utility functions
import io.github.tailhq.dynaml.utils
//Kernels for GP,SVM models
import io.github.tailhq.dynaml.kernels._
//Shell examples
import io.github.tailhq.dynaml.examples._
//Load neural net primitives
import io.github.tailhq.dynaml.models.neuralnets.TransferFunctions._
//The probability API
import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.probability.distributions._
//Wavelet API
import io.github.tailhq.dynaml.wavelets._
//OpenML support
import io.github.tailhq.dynaml.openml.OpenML
//Spark support
import io.github.tailhq.dynaml.DynaMLSpark._
//Renjin imports
import javax.script._
//TensorFlow imports
import _root_.io.github.tailhq.dynaml.tensorflow._
import _root_.io.github.tailhq.dynaml.tensorflow.implicits._
import org.renjin.script._
import org.renjin.sexp._
val r_engine_factory = new RenjinScriptEngineFactory()
implicit val renjin = r_engine_factory.getScriptEngine()
val r: String => SEXP = (s: String) => renjin.eval(s).asInstanceOf[SEXP]
val R: java.io.File => Unit = (f: java.io.File) => renjin.eval(f)
