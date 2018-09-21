package io.github.mandar2812.dynaml.graphics.plot3d

import org.jzy3d.analysis.AbstractAnalysis
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.maths.Coord3d
import org.jzy3d.maths.algorithms.interpolation.algorithms.BernsteinInterpolator
import org.jzy3d.plot3d.primitives.{LineStripInterpolated, Shape}
import org.jzy3d.plot3d.rendering.canvas.Quality
import scala.collection.JavaConverters._


class LinePlot3D(points: Traversable[(Float, Float, Float)], resolution: Int, drawPoints: Boolean)
  extends AbstractAnalysis {

  override def init(): Unit = {

    val linePlot3D = new LineStripInterpolated(
      new BernsteinInterpolator,
      points.map(p => new Coord3d(p._1, p._2, p._3)).toList.asJava,
      resolution, drawPoints
    )

    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(linePlot3D)

  }
}
