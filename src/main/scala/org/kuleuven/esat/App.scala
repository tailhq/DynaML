package org.kuleuven.esat

import java.io.File
import breeze.linalg.{norm, DenseVector}
import com.github.tototoshi.csv._
import com.tinkerpop.blueprints.pgm.impls.tg.{TinkerGraphFactory, TinkerGraph}
import com.tinkerpop.gremlin.scala._

/**
 * Hello world!
 *
 */
object App extends App {

  override def main(args: Array[String]): Unit = {
    //Read csv file

    val g = TinkerGraphFactory.createTinkerGraph()
    var delim: Char = ','
    if(args.apply(1).compare("tab") == 0) delim = '\t'

    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = delim
      override val quoting = QUOTE_NONNUMERIC
    }

    val reader = CSVReader.open(new File(args.apply(0)))

    val head: Boolean = true
    val lines = reader.iterator
    var index = 1
    var dim = 0
    if(head) {
      dim = lines.next().length
    }

    g.addVertex("w").setProperty("variable", "parameter")
    g.getVertex("w").setProperty("slope", DenseVector.ones[Double](dim))


    while (lines.hasNext) {
      //Parse line and extract features
      val line = lines.next()
      val yv = line.apply(line.length - 1).toDouble
      val features = line.map((s) => s.toDouble).toArray
      features.update(line.length - 1, 1.0)
      val xv: DenseVector[Double] =
        new DenseVector[Double](features)

      /*
      * Create nodes xi and yi
      * append to them their values
      * properties, etc
      * */

      g.addVertex(("x", index)).setProperty("value", xv)
      g.getVertex(("x", index)).setProperty("variable", "data")

      g.addVertex(("y", index)).setProperty("value", yv)
      g.getVertex(("y", index)).setProperty("variable", "target")

      //Add edge between xi and yi
      g.addEdge((("x", index), ("y", index)),
        g.getVertex(("x", index)), g.getVertex(("y", index)),
        "causes")

      //Add edge between w and y_i
      g.addEdge(("w", ("y", index)), g.getVertex("w"),
        g.getVertex(("y", index)),
        "controls")

      index += 1
    }


    val g1 = ScalaGraph.wrap(g)

    val learingRate = 0.001
    var count = 1
    var w = ScalaVertex.wrap(g1.getVertex("w"))
    var oldW: DenseVector[Double] = w.getProperty("slope").asInstanceOf[DenseVector[Double]]
    var newW: DenseVector[Double] = DenseVector.zeros(4)
    var diff: DenseVector[Double] = DenseVector.zeros(4)

    while(count <= args.apply(2).toInt) {
      //println("************ Iteration "+count+" ************")
      val targets = w.getOutEdges().iterator()
      while (targets.hasNext) {
        w = g1.getVertex("w")
        oldW = w.getProperty("slope").asInstanceOf[DenseVector[Double]]
        val edge = ScalaEdge.wrap(targets.next())
        val yV = ScalaVertex.wrap(edge.getInVertex)
        val y = yV.getProperty("value").asInstanceOf[Double]

        val xV = yV.getInEdges("causes").iterator().next().getOutVertex
        val x = xV.getProperty("value").asInstanceOf[DenseVector[Double]]
        //println("x "+x.toString())
        //println("y "+y.toString())

        val dt: Double = oldW dot x
        val grad: Double = (dt - y)/index.toDouble
        //println("grad " + grad)
        diff = DenseVector.tabulate(oldW.length)((i) =>
          2*learingRate*((x(i) * grad) + oldW(i)))
        newW = oldW - diff
        g1.getVertex("w").setProperty("slope", newW)
      }
      //println("old "+oldW)
      //println("new "+newW)
      count += 1
    }

    println(g1.getVertex("w").getProperty("slope")
      .asInstanceOf[DenseVector[Double]])
  }

}
