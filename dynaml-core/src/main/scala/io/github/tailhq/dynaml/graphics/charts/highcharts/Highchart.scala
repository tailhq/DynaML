package io.github.tailhq.dynaml.graphics.charts.highcharts

import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import scala.collection._
import scala.language.implicitConversions

/**
 * User: austin
 * Date: 9/9/13
 *
 * Tries to closely follow : api.highcharts.com/highcharts
 *
 * Original built for Highcharts 3.0.6, and we are now porting to 4.0.4 (12/12/14)
 */

/**
 * implicits to offer conversions from scala types to Highcharts objects.
 * Including wrappers around Option, and transformations from Traversable / Array to Highcharts series
 */
object Highchart {
  // Data
  implicit def traversableToTraversableData[X: Numeric, Y: Numeric](seq: Traversable[(X, Y)]) = seq.map{case(x, y) => Data(x, y)}

  // Series
  implicit def traversableToTraversableSeries[X: Numeric, Y: Numeric](seq: Traversable[(X, Y)]) = seriesToTraversableSeries(traversableToSeries(seq))
  implicit def traversableToSeries[X: Numeric, Y: Numeric](seq: Traversable[(X, Y)]) = Series(traversableToTraversableData(seq))
  implicit def seriesToTraversableSeries(series: Series) = Seq(series)
  implicit def traversableToSomeArray(t: Traversable[Any]) = Some(t.toArray) // for axes

  // Axis
  implicit def axisTitleOptionToArrayAxes(axisTitle: Option[AxisTitle]) = Some(Array(Axis(axisTitle)))
  implicit def axisToArrayAxes(axis: Axis) = Some(Array(axis))
  implicit def axesSeqToSomeAxesArray(axes: Seq[Axis]) = Some(axes.toArray)
  implicit def stringToAxisTitle(s: String) = Some(AxisTitle(s))
  implicit def stringToAxis(s: String): Option[Array[Axis]] = axisTitleOptionToArrayAxes(stringToAxisTitle(s))

  // Colors
  implicit def colorToSomeColorArray(c: Color.Type) = Some(Array(c))

  // Exporting
  implicit def stringToExporting(s: String) = Some(Exporting(s))

  // Title
  implicit def stringToTitle(s: String) = Some(Title(text = s))

  // value -> Some(value)
  implicit def optionWrap[T](value: T): Option[T] = Option(value)
}

/**
 * Ensures the object can be cast to Map[String, Any] so we can perform json serialization
 * @param _name
 */
abstract class HighchartKey(var _name: String) {
  def toServiceFormat: Map[String, Any]
}

//Internal functions for mapping to service format (which gets parsed to json)
object HighchartKey {
  def highchartKeyToServiceFormat(hck: HighchartKey): Map[String, Any] = Map(hck._name -> hck.toServiceFormat)

  def optionToServiceFormat(o: Option[HighchartKey]): Map[String, Any] = o match {
    case None => Map()
    case Some(s) => highchartKeyToServiceFormat(s)
  }

  def optionArrayAxisToServiceFormat(o: Option[Array[Axis]]): Map[String, Any] = o match {
    case None => Map()
    case Some(s) => Map(s.head.__name -> s.map(_.toServiceFormat))
  }

  def optionArrayColorToServiceFormat(o: Option[Array[Color.Type]]): Map[String, Any] = o match {
    case None => Map()
    case Some(a) => Map("colors" -> a)
  }

  def axisToTitleId(a: Axis) = a.id

  def someAxisToTitleId(oa: Option[Axis]) = oa match {
    case None => "0"
    case Some(a) => a.id
  }

  def hckTraversableToServiceFormat(t: Traversable[HighchartKey]): Map[String, Any] = {
    if(t.isEmpty) Map()
    else Map(t.head._name -> t.map(_.toServiceFormat))
  }

  def flatten(o: (String, Option[Any])) = o._2 match {
    case None => None
    case Some(v) => Some(o._1, v)
  }

  def someStyleToServiceFormat(style: Option[CSSObject]) =
  {if (style != None) Map("style" -> style.get.toString()) else Map()}
}

// Not going to implement: loading
// Not done yet:
//  css object wrappers limited support
//  navigation :: (styling for exporting)
//  pane: for guages (where are guages?)
//
/**
 * Top-most level Highcharts object. Overrides some of the Highcharts defaults
 * @param series
 * @param title
 * @param chart
 * @param colors
 * @param credits
 * @param exporting
 * @param legend
 * @param plotOptions
 * @param subtitle
 * @param setTurboThreshold
 * @param tooltip
 * @param xAxis
 * @param yAxis
 */
case class Highchart(
                      series: Traversable[Series],
                      title: Option[Title] = Some(Title()),
                      chart: Option[Chart] = Some(Chart()),
                      colors: Option[Array[Color.Type]] = None,
                      credits: Option[Credits] = Some(Credits()),
                      exporting: Option[Exporting] = Some(Exporting()),
                      legend: Option[Legend] = None,
                      plotOptions: Option[PlotOptions] = Some(PlotOptions()),
                      subtitle: Option[Title] = None,
                      setTurboThreshold: Boolean = true,
                      tooltip: Option[ToolTip] = None,
                      xAxis: Option[Array[Axis]] = Some(Array(Axis())),
                      yAxis: Option[Array[Axis]] = Some(Array(Axis()))
                      ) {

  // Stylistic API
  // Assigns a label to the axis
  import Highchart._

  // Axis Labels
  def xAxis(label: String): Highchart = {
    this.copy(xAxis = Some(xAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map { _.copy(title = label) }
      case _ => Array(Axis(AxisTitle(label)))
    }))
  }
  def yAxis(label: String): Highchart = {
    this.copy(yAxis = Some(yAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map { _.copy(title = label) }
      case _ => Array(Axis(AxisTitle(label)))
    }))
  }

  // Change the type of the axis, ie logarithmic
  // To convert to categories prefer xAxisCategories
  def xAxisType(axisType: AxisType.Type): Highchart = {
    if (!AxisType.values.contains(axisType)) {
      println(s"Not an acceptable axis type. Options are: ${AxisType.values.mkString(", ")}.")
      return this
    }
    this.copy(xAxis = Some(xAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map { _.copy(axisType = axisType) }
      case _ => Array(Axis(axisType = axisType))
    }))
  }
  def yAxisType(axisType: AxisType.Type): Highchart = {
    if (!AxisType.values.contains(axisType)) {
      println(s"Not an acceptable axis type. Options are: ${AxisType.values.mkString(", ")}.")
      return this
    }
    this.copy(yAxis = Some(yAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map { _.copy(axisType = axisType) }
      case _ => Array(Axis(axisType = axisType))
    }))
  }

  // Modifies the axis to use String based category names instead of a numeric series
  def xAxisCategories(categories: Iterable[String]): Highchart = {
    this.copy(xAxis = Some(xAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map{ _.copy(
        axisType = AxisType.category,
        categories = Some(categories.toArray)
      )}
      case _ => Array(Axis(
        axisType = AxisType.category,
        categories = Some(categories.toArray)
      ))
    }))
  }
  def yAxisCategories(categories: Iterable[String]): Highchart = {
    this.copy(yAxis = Some(yAxis match {
      case Some(axisArray) if axisArray.size > 0 => axisArray.map{ _.copy(
        axisType = AxisType.category,
        categories = Some(categories.toArray)
      )}
      case _ => Array(Axis(
        axisType = AxisType.category,
        categories = Some(categories.toArray)
      ))
    }))
  }

  // Update the title at the top of the chart
  def title(label: String): Highchart = {
    this.copy(title = label)
  }

  // Assign names to series, if mis-matched lengths use the shorter one as a cut-off
  def legend(labels: Iterable[String]): Highchart = {
    val labelArray = labels.toArray
    val newSeries = series.toSeq.zipWithIndex.map { case (s, idx) => if (idx >= labels.size) s else s.copy(name = Some(labelArray(idx))) }
    this.copy(series = newSeries)
  }

  // Combines points with the ame x-value into a single visualization point
  // normal stacking adds the values in order of the corresponding series
  // percentage stacking creates a distribution from the values
  def stack(stackType: Stacking.Type): Highchart = {
    this.copy(plotOptions = Some(PlotOptions(series = PlotOptionKey(stacking = stackType))))
  }

  // Undoes the effect of calling stack()
  def unstack(): Highchart = {
    this.copy(plotOptions = Some(PlotOptions(series = PlotOptionKey(stacking = None))))
  }

  // Json manipulation

  import HighchartKey._

  implicit val formats = Serialization.formats(NoTypeHints)
  def toJson = Serialization.write(jsonMap)

  def jsonMap: Map[String, Any] = {
    if(series.size == 0) System.err.println("Warning: created a plot with no data - visualization will be empty")

    // Because we want to default to turboThreshold off (0) we control it as a boolean at the top-most level
    // As otherwise it is a per-type plotOption
    val turboOutput: Map[String, Map[String, Any]] =
      if(setTurboThreshold) {
        series.filter(_.chart != Some(SeriesType.pie)).flatMap(_.chart).map { s =>
          s -> Map("turboThreshold" -> 0)
        }.toMap
      } else Map.empty[String, Map[String, Any]]

    // Todo: can we do better?
    // Check if it exists
    val finalPlotOption = if(plotOptions.isDefined) {
      val plotOptionsMap = optionToServiceFormat(plotOptions)(PlotOptions.name).asInstanceOf[Map[String, Map[String, Any]]]
      val keys = plotOptionsMap.keySet ++ turboOutput.keySet
      val combinedMap = keys.map{key => key -> (plotOptionsMap.getOrElse(key, Map()) ++ turboOutput.getOrElse(key, Map()))}.toMap
      Map(PlotOptions.name -> combinedMap).asInstanceOf[Map[String, Any]]
    } else {
      Map(PlotOptions.name -> turboOutput)
    }

    val colorWrapper = (colors, yAxis) match {
      case (Some(c), _) if c.size > 0 => colors
      case (_, Some(y)) => {
        val styleColors = y.flatMap(_.title).flatMap(_.style).flatMap(_.color)
        if(styleColors.size == series.size) Some(styleColors)
        else None
      }
      case _ => None
    }

    (yAxis, series) match {
      case (Some(y), s) => if(y.size > 1 && y.size == s.size) y.zip(s.toSeq).zipWithIndex.foreach{case((axis, ser), index) =>
        if(axis.id.isEmpty) axis.id = Some(index.toString())
        if(ser.yAxis.isEmpty) ser.yAxis = axis.id
      }
      case _ =>
    }

    (xAxis, series) match {
      case (Some(x), s) => if(x.size > 1 && x.size == s.size) x.zip(s.toSeq).zipWithIndex.foreach{case((axis, ser), index) =>
        if(axis.id.isEmpty) axis.id = Some(index.toString())
        if(ser.xAxis.isEmpty) ser.xAxis = axis.id
      }
      case _ =>
    }

    // Axis defaults to yAxis, rename xAxes
    xAxis.map(_.foreach(_.__name = "xAxis"))

      finalPlotOption ++
        hckTraversableToServiceFormat(series) ++
        Seq(xAxis, yAxis).flatMap(optionArrayAxisToServiceFormat) ++
        optionArrayColorToServiceFormat(colorWrapper) ++
        Seq(chart, title, exporting, credits, legend, tooltip, subtitle).flatMap(optionToServiceFormat)
  }

  def toServiceFormat: (String, Map[String, Any]) = {
     "highcharts" -> jsonMap
  }
}

// can we do better than specifying every field manually? (probably...)
// but I was not happy with Enumeration returning type Value instead of type String
// I need to look into jerkson or something similar for case class -> json conversion
case class Title(
                  text: String = "", // Override default "Chart title"
                  align: Option[Alignment.Type] = None,
                  floating: Option[Boolean] = None,
                  style: Option[CSSObject] = None,
                  useHTML: Option[Boolean] = None,
                  verticalAlign: Option[VerticalAlignment.Type] = None,
                  x: Option[Int] = None,
                  y: Option[Int] = None,
                  var __name: String = "title"
                  ) extends HighchartKey(__name) {
  def toServiceFormat =
    Map("text" -> text) ++
      Map(
        "align" -> align,
        "floating" -> floating,
        "useHTML" -> useHTML,
        "verticalAlign" -> verticalAlign,
        "x" -> x,
        "y" -> y
      ).flatMap(HighchartKey.flatten)  ++
      HighchartKey.someStyleToServiceFormat(style)
}

case class Chart(
                  // todo, many other chart options
                  zoomType: Option[Zoom.Type] = Some(Zoom.xy)
                  ) extends HighchartKey("chart") {
  def toServiceFormat = Map(
    "zoomType" -> zoomType
  ).flatMap(HighchartKey.flatten)
}

case class Credits(
                    enabled: Option[Boolean] = None,
                    href: String = "", // Override default Highcharts
                    position: Option[Position] = None,
                    style: Option[CSSObject] = None,
                    text: String = "" // Override default Highcharts
                    ) extends HighchartKey("credits") {
  def toServiceFormat = Map(
    "href" -> href,
    "text" -> text
  ) ++
    Map("style" -> style, "enabled" -> enabled).flatMap(HighchartKey.flatten) ++
    HighchartKey.optionToServiceFormat(position) ++
    HighchartKey.someStyleToServiceFormat(style)}

case class Exporting(
                      //buttons
                      //chartOptions
                      filename: String = "chart",
                      scale: Option[Int] = None,
                      sourceHeight: Option[Int] = None,
                      sourceWidth: Option[Int] = None,
                      _type: Option[String] = None,
                      url: Option[String] = None,
                      width: Option[Int] = None
                      ) extends HighchartKey("exporting") {

  def toServiceFormat =
    Map("filename" -> filename) ++
      Map(
        "scale" -> scale,
        "type" -> _type,
        "url" -> url,
        "sourceHeight" -> sourceHeight,
        "sourceWidth" -> sourceWidth,
        "width" -> width
      ).flatMap(HighchartKey.flatten)
}

case class Position(
                     align: Option[Alignment.Type] = None,
                     x: Option[Int] = None,
                     verticalAlign: Option[VerticalAlignment.Type] = None,
                     y: Option[Int] = None
                     ) extends HighchartKey("position") {
  def toServiceFormat = Map(
    "align" -> align,
    "x" -> x,
    "verticalAlign" -> verticalAlign,
    "y" -> y
  ).flatMap(HighchartKey.flatten)
}

case class ToolTip(
                    animation: Option[Boolean] = None,
                    backgroundColor: Option[Color.Type] = None,
                    borderColor: Option[Color.Type] = None,
                    borderRadius: Option[Int] = None,
                    borderWidth: Option[Int] = None,
                    // crosshairs
                    dateTimeLabelFormats: Option[DateTimeFormats] = None, // this has different defaults than the Axis
                    enabled: Option[Boolean] = None,
                    followPointer: Option[Boolean] = None,
                    followTouchMove: Option[Boolean] = None,
                    footerFormat: Option[String] = None,
                    //formatter
                    //headerFormat
                    hideDelay: Option[Int] = None,
                    //pointFormat
                    //positioner
                    shadow: Option[Boolean] = None,
                    shared: Option[Boolean] = None,
                    snap: Option[Int] = None,
                    //style
                    useHTML: Option[Boolean] = None,
                    valueDecimals: Option[Int] = None,
                    valuePrefix: Option[String] = None,
                    valueSuffix: Option[String] = None,
                    xDateFormat: Option[String] = None
                    ) extends HighchartKey("tooltip") {

  def toServiceFormat =
    Map(
      "animation" -> animation,
      "backgroundColor" -> backgroundColor,
      "borderColor" -> borderColor,
      "borderRadius" -> borderRadius,
      "borderWidth" -> borderWidth,
      "enabled" -> enabled,
      "followPointer" -> followPointer,
      "followTouchMove" -> followTouchMove,
      "footerFormat" -> footerFormat,
      "hideDelay" -> hideDelay,
      "shadow" -> shadow,
      "shared" -> shared,
      "snap" -> snap,
      "useHTML" -> useHTML,
      "valueDecimals" -> valueDecimals,
      "valuePrefix" -> valuePrefix,
      "valueSuffix" -> valueSuffix,
      "xDateFormat" -> xDateFormat
    ).flatMap(HighchartKey.flatten) ++
      HighchartKey.optionToServiceFormat(dateTimeLabelFormats)

}

case class Series(
                    data: Traversable[Data[_, _]],
                    index: Option[Int] = None,
                    legendIndex: Option[Int] = None,
                    name: Option[String] = None,
                    chart: Option[SeriesType.Type] = Some(SeriesType.line), // for turbo threshold default
                    visible: Option[Boolean] = None,
                    color: Option[Color.Type] = None,
                    var xAxis: Option[String] = None,
                    var yAxis: Option[String] = None,
                    __name: String = "series"
) extends HighchartKey(__name) {

  def toServiceFormat: Map[String, Any] = {
    if (data.size == 0) System.err.println("Tried to create a series with no data")
    Map("data" -> data.map(_.toServiceFormat).toSeq) ++
    Map("xAxis" -> xAxis, "yAxis" -> yAxis, "type" -> chart, "color" -> color, "visible" -> visible, " index" -> index, "legendIndex" -> legendIndex, "name" -> name).flatMap{HighchartKey.flatten}
  }
}

trait Data[X, Y] {
  def toServiceFormat: Map[String, Any]
}

object Data {
  def apply[X: Numeric, Y: Numeric](
                                     x: X,
                                     y: Y,
                                     color: Option[Color.Type] = None,
                                     //dataLabels
                                     //events
                                     // id
                                     name: Option[String] = None
                                     ): DataPair[X, Y] = DataPair(x, y, color, name)

  def apply[T: Numeric](
                         x: Any, // TODO x type
                         low: T,
                         q1: T,
                         median: T,
                         q3: T,
                         high: T
                         // todo all the other options...
                         ): BoxplotData[T] = BoxplotData(Some(x), low, q1, median, q3, high)

  def apply[T: Numeric](
                         low: T,
                         q1: T,
                         median: T,
                         q3: T,
                         high: T
                         // todo all the other options...
                         ): BoxplotData[T] = BoxplotData(None, low, q1, median, q3, high)
}

// Most series take in data points as (x, y)
case class DataPair[X: Numeric, Y: Numeric](
                                             x: X,
                                             y: Y,
                                             color: Option[Color.Type] = None,
                                             //dataLabels
                                             //events
                                             // id
                                             name: Option[String] = None
                                             ) extends Data[X, Y] {

  def toServiceFormat = {
    Map("x" -> x, "y" -> y) ++
      Map("color" -> color, "name" -> name).flatMap{HighchartKey.flatten}
  }
}

// Box plot takes in data as an array of size five: lower-whisker, lower-box, median, upper-box, upper-whisker
case class BoxplotData[T: Numeric](
                                x: Option[Any], // TODO x type
                                low: T,
                                q1: T,
                                median: T,
                                q3: T,
                                high: T
                                // todo all the other options...
                                ) extends Data[T, T] {
  def toServiceFormat = {
    Map("low" -> low, "q1" -> q1, "median" -> median, "q3" -> q3, "high" -> high) ++
      Map("x" -> x).flatMap{HighchartKey.flatten}
  }
}

// TODO PieData for legendIndex, slice

// No more than 22 members in a case class TODO
case class Legend(
                   align: Option[Alignment.Type] = None,
                   backgroundColor: Option[Color.Type] = None,
                   borderColor: Option[Color.Type] = None,
                   //  borderRadius: Int = 5,
                   //  borderWidth: Int = 2,
                   enabled: Option[Boolean] = Some(false), // override default
                   floating: Option[Boolean] = None,
                   itemDistance: Option[Int] = None,
                   //itemHiddenStyle
                   //itemHoverStyle
                   itemMarginBottom: Option[Int] = None,
                   itemMarginTop: Option[Int] = None,
                   //itemStyle
                   itemWidth: Option[Int] = None,
                   labelFormat: Option[String] = None, // TODO - format string helpers
                   //labelFormatter
                   layout: Option[Layout.Type] = None,
                   margin: Option[Int] = None,
                   maxHeight: Option[Int] = None,
                   //navigation
                   padding: Option[Int] = None,
                   reversed: Option[Boolean] = None,
                   rtl: Option[Boolean] = None, // right-to-left
                   //shadow
                   //style
                   //  symbolPadding: Int = 5,
                   //  symbolWidth: Int = 30,
                   title: Option[String] = None, // todo - css title
                   //  useHTML: Boolean = false,
                   verticalAlign: Option[VerticalAlignment.Type] = None,
                   width: Option[Int] = None,
                   x: Option[Int] = None,
                   y: Option[Int] = None
                   ) extends HighchartKey("legend") {

  def toServiceFormat =
    Map(
      "align" -> align,
      "backgroundColor" -> backgroundColor,
      "borderColor" -> borderColor,
      //      "borderRadius" -> borderRadius,
      //      "borderWidth" -> borderWidth,
      "enabled" -> enabled,
      "floating" -> floating,
      "itemDistance" -> itemDistance,
      "itemMarginBottom" -> itemMarginBottom,
      "itemMarginTop" -> itemMarginTop,
      "labelFormat" -> labelFormat,
      "layout" -> layout,
      "margin" -> margin,
      "padding" -> padding,
      "reversed" -> reversed,
      "rtl" -> rtl,
      //      "symbolPadding" -> symbolPadding,
      //      "symbolWidth" -> symbolWidth,
      //      "useHTML" -> useHTML,
      "verticalAlign" -> verticalAlign,
      "x" -> x,
      "y" -> y,
      "itemWidth" -> itemWidth,
      "maxHeight" -> maxHeight,
      "title" -> title,
      "width" -> width
    ).flatMap{case(s, a) => HighchartKey.flatten(s, a)}
}

case class Axis(
                 title: Option[AxisTitle] = Some(AxisTitle()),
                 allowDecimals: Option[Boolean] = None,
                 alternateGridColor: Option[Color.Type] = None,
                 categories: Option[Array[String]] = None,
                 dateTimeLabelFormats: Option[DateTimeFormats] = None,
                 endOnTick: Option[Boolean] = None,
                 //events
                 //  gridLineColor: Color.Type = "#C0C0C0",
                 //  gridLineDashStyle: String = "Solid",       // TODO Bundle
                 //  gridLineWidth: Int = 2,
                 var id: Option[String] = None,
                 labels: Option[AxisLabel] = None,
                 lineColor: Option[Color.Type] = None,
                 lineWidth: Option[Int] = None,
                 //linkedTo
                 max: Option[Int] = None,
                 //  maxPadding: Double = 0.01,
                 min: Option[Int] = None,
                 //  minPadding: Double = 0.01,
                 minRange: Option[Int] = None,
                 minTickInterval: Option[Int] = None,
                 //minor
                 offset: Option[Int] = None,
                 opposite: Option[Boolean] = None, // opposite side of chart, ie left / right for y-axis
                 //plotBands
                 //plotLines // TODO Kevin wants these
                 reversed: Option[Boolean] = None,
                 //  showEmpty: Boolean = false,
                 showFirstLabel: Option[Boolean] = None,
                 showLastLabel: Option[Boolean] = None,
                 //startOfWeek
                 startOnTick: Option[Boolean] = None,
                 //  tickColor: Color.Type = "#C0D0E0",
                 // TICK STUFF TODO
                 axisType: Option[AxisType.Type] = None,
                 var __name: String = "yAxis"
                 ) extends HighchartKey(__name) {

  def toServiceFormat: Map[String, Any] =
    Map(
      "allowDecimals" -> allowDecimals,
      "categories" -> categories,
      "endOnTick" -> endOnTick,
      "lineColor" -> lineColor,
      "lineWidth" -> lineWidth,
      //    "maxPadding" -> maxPadding,
      //    "minPadding" -> minPadding,
      "max" -> max,
      "min" -> min,
      "offset" -> offset,
      "opposite" -> opposite,
      "reversed" -> reversed,
      "showFirstLabel" -> showFirstLabel,
      "showLastLabel" -> showLastLabel,
      "startOnTick" -> startOnTick,
      "type" -> axisType,
      "title" -> title,
      "id" -> id
    ).flatMap(HighchartKey.flatten) ++
      HighchartKey.optionToServiceFormat(dateTimeLabelFormats) ++
      HighchartKey.optionToServiceFormat(labels)
}

case class AxisLabel(
                      align: Option[String] = None,
                      enabled: Option[Boolean] = None,
                      format: Option[String] = None,
                      //                            formatter
                      maxStaggerLines: Option[Int] = None,
                      overflow: Option[Overflow.Type] = None,
                      rotation: Option[Int] = None,
                      step: Option[Int] = None,
                      style: Option[CSSObject] = None,
                      useHTML: Option[Boolean] = None,
                      x: Option[Int] = None,
                      y: Option[Int] = None,
                      zIndex: Option[Int] = None
                      ) extends HighchartKey("labels") {
  def toServiceFormat =
    Map(
      "align" -> align,
      "enabled" -> enabled,
      "format" -> format,
      "maxStaggerLines" -> maxStaggerLines,
      "overflow" -> overflow,
      "rotation" -> rotation,
      "step" -> step,
      "useHTML" -> useHTML,
      "x" -> x,
      "y" -> y,
      "zIndex" -> zIndex
    ).flatMap(HighchartKey.flatten) ++
      HighchartKey.someStyleToServiceFormat(style)
}

case class DateTimeFormats(
                            millisecond: String = "%H:%M:%S.%L",
                            second: String = "%H:%M:%S",
                            minute: String = "%H:%M",
                            hour: String = "%H:%M",
                            day: String = "%e. %b",
                            week: String = "%e. b",
                            month: String = "%b \\ %y",
                            year: String = "%Y"
                            ) extends HighchartKey("dateTimeLabelFormats") {

  def toServiceFormat = Map("dateTimeLabelFormats" ->
    Map(
      "millisecond" -> millisecond,
      "second" -> second,
      "minute" -> minute,
      "hour" -> hour,
      "day" -> day,
      "week" -> week,
      "month" -> month,
      "year" -> year
    )
  )
}

// Must supply text, others default to align=middle, maring=(x=0, y=10), offset=(relative), rotation=0
case class AxisTitle(
                      text: String = "", // Override default y-axis "value"
                      align: Option[AxisAlignment.Type] = None,
                      margin: Option[Int] = None,
                      offset: Option[Int] = None,
                      rotation: Option[Int] = None,
                      style: Option[CSSObject] = None
                      ) {
  def toServiceFormat =
    Map("text" -> text) ++
      Map("align" -> align, "margin" -> margin, "offset" -> offset, "rotation" -> rotation).flatMap(HighchartKey.flatten) ++
      HighchartKey.someStyleToServiceFormat(style)
}

object AxisTitle {
  def apply(text: String, color: Color.Type) =
    new AxisTitle(text, style = Some(CSSObject(Some(color))))

  def apply(text: String, color: Color.Type, rotation: Option[Int]) =
    new AxisTitle(text, rotation = rotation, style = Some(CSSObject(Some(color))))
}

