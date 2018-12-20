package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseMatrix, NumericOps}
import breeze.linalg._
import breeze.linalg.operators.OpSolveMatrixBy
import org.apache.spark.rdd.RDD

import scala.collection.immutable.NumericRange

import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._

/**
  * @author mandar2812 date 13/10/2016.
  * A distributed matrix that is stored in blocks.
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a breeze [[DenseMatrix]] containing
  *             all the elements in the said block.
  */
private[dynaml] class PartitionedMatrix(
  data: Stream[((Long, Long), DenseMatrix[Double])],
  num_rows: Long = -1L, num_cols: Long = -1L,
  num_row_blocks: Long = -1L, num_col_blocks: Long = -1L)
  extends AbstractPartitionedMatrix[DenseMatrix[Double]](data, num_row_blocks, num_col_blocks)
    with NumericOps[PartitionedMatrix] {

  lazy val rows: Long = if(num_rows == -1L) data.filter(_._1._2 == 0L).map(_._2.rows).sum.toLong else num_rows

  lazy val cols: Long = if(num_cols == -1L) data.filter(_._1._1 == 0L).map(_._2.cols).sum.toLong else num_cols

  override def repr: PartitionedMatrix = this

  /**
    * Transpose of blocked matrix
    * */
  override def t: PartitionedMatrix = new PartitionedMatrix(
    _data.map(c => (c._1.swap, c._2.t)),
    cols, rows, colBlocks, rowBlocks)

  /**
    * Map/transform each block matrix.
    * @param f The mapping function
    */
  override def map(
    f: (((Long, Long), DenseMatrix[Double])) => ((Long, Long), DenseMatrix[Double])): PartitionedMatrix =
    new PartitionedMatrix(data.map(f), rows, cols, rowBlocks, colBlocks)

  /**
    * Slice a blocked matrix to produce a new block matrix.
    */
  def apply(r: NumericRange[Long], c: NumericRange[Long]): PartitionedMatrix =
    new PartitionedMatrix(
      data.filter(e => r.contains(e._1._1) && c.contains(e._1._2))
        .map(e => ((e._1._1 - r.min, e._1._2 - c.min), e._2)),
      num_row_blocks = r.length, num_col_blocks = c.length
    )


  /**
    * Get lower triangular portion of matrix
    */
  def L: LowerTriPartitionedMatrix = {
    require(rows == cols, "Matrix must be square for lower triangular component to make sense")
    require(rowBlocks == colBlocks,
      "Matrix must be uniformly partitioned for lower triangular component to be efficiently computed")
    new LowerTriPartitionedMatrix(
      filterBlocks(c => c._1 <= c._2)
        .map(bl =>
          if(bl._1._1 == bl._1._2) (bl._1, lowerTriangular(bl._2))
          else bl), rows, cols, rowBlocks, colBlocks)
  }

  /**
    * Upper triangular portion of matrix
    */
  def U: UpperTriPartitionedMatrix = {
    require(rows == cols, "Matrix must be square for lower triangular component to make sense")
    require(rowBlocks == colBlocks,
      "Matrix must be uniformly partitioned for lower triangular component to be efficiently computed")
    new UpperTriPartitionedMatrix(
      filterBlocks(c => c._1 >= c._2)
        .map(bl =>
          if(bl._1._1 == bl._1._2) (bl._1, upperTriangular(bl._2))
          else bl), rows, cols, rowBlocks, colBlocks)
  }

  /**
    * Convert to breeze matrix. NOTE: do not use
    * on large block matrices as it would cause JVM
    * memory overflow
    */
  def toBreezeMatrix =
    DenseMatrix.vertcat[Double](
      _data.groupBy(_._1._1).map(row => DenseMatrix.horzcat(row._2.sortBy(_._1._2).map(_._2):_*)).toSeq:_*
    )

}


object PartitionedMatrix {

  /**
    * Construct a partitioned matrix from a stream of matrix blocks.
    * */
  def apply(d: Stream[((Long, Long), DenseMatrix[Double])], numrows: Long, numcols: Long): PartitionedMatrix =
    new PartitionedMatrix(d, num_rows = numrows, num_cols = numcols)

  /**
    * Construct a partitioned matrix using a tabulation function.
    * */
  def apply(nRows: Long, nCols: Long,
            numElementsPerRBlock: Int, numElementsPerCBlock: Int,
            tabFunc: (Long, Long) => Double) = {
    val nRblocks = math.ceil(nRows.toDouble/numElementsPerRBlock).toLong
    val nCblocks = math.ceil(nCols.toDouble/numElementsPerCBlock).toLong

    val resRows: Int = if(nRows%numElementsPerRBlock == 0) numElementsPerRBlock
    else (nRows%numElementsPerRBlock).toInt

    val resColumns: Int = if(nCols%numElementsPerCBlock == 0) numElementsPerCBlock
    else (nCols%numElementsPerCBlock).toInt

    val blockIndices = for(i <- 0L until nRblocks; j <- 0L until nCblocks) yield (i,j)

    val bMat = blockIndices.sorted.toStream.map(c => {

      val matDimensions =
        if(c._1 < nRblocks-1L && c._2 < nCblocks - 1L) (numElementsPerRBlock, numElementsPerCBlock)
        else if(c._1 == nRblocks-1L && c._2 < nCblocks - 1L) (resRows, numElementsPerCBlock)
        else if(c._1 < nRblocks-1L && c._2 == nCblocks - 1L) (numElementsPerRBlock, resColumns)
        else (resRows, resColumns)

      (c, DenseMatrix.tabulate[Double](matDimensions._1, matDimensions._2)(
        (i,j) => tabFunc(
          i.toLong + c._1*numElementsPerRBlock,
          j.toLong + c._2*numElementsPerCBlock)
      ))

    })

    new PartitionedMatrix(
      bMat, num_rows = nRows, num_cols = nCols,
      num_row_blocks = nRblocks, num_col_blocks = nCblocks)

  }

  /**
    * Vertically concatenate partitioned matrices
    * */
  def vertcat(vectors: PartitionedMatrix*): PartitionedMatrix = {
    //sanity check
    require(vectors.map(_.colBlocks).distinct.length == 1,
      "In case of vertical concatenation of matrices their columns sizes must be equal")

    val sizes = vectors.map(_.rowBlocks)
    new PartitionedMatrix(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._data.map(c => ((c._1._1+offset, c._1._2), c._2))
    }).reduce((a,b) => a.union(b)).sortBy(_._1))
  }


 /**
   * Horizontally concatenate a collection of breeze [[DenseVector]] instances
   * to form a [[PartitionedMatrix]]
   *
   * */
  def horzcat(nC: Int)(vectors: DenseVector[Double]*): PartitionedMatrix = {

    val num_rows = vectors.head.length.toLong
    val num_cols = vectors.length.toLong
    apply(
      vectors.grouped(nC).zipWithIndex.map(c => ((0L, c._2.toLong), DenseMatrix.horzcat(c._1.map(_.toDenseMatrix):_*))).toStream,
      num_rows, num_cols)
  }


}

/**
  * A partitioned lower triangular matrix.
  *
  * @author mandar2812 date: 18/10/2016
  * */
private[dynaml] class LowerTriPartitionedMatrix(
  underlyingdata: Stream[((Long, Long), DenseMatrix[Double])],
  num_rows: Long = -1L, num_cols: Long = -1L,
  num_row_blocks: Long = -1L, num_col_blocks: Long = -1L)
  extends PartitionedMatrix(
    data = underlyingdata
      .map(c =>
        if(c._1._1 == c._1._2) Seq(c)
        else Seq(c, (c._1.swap, DenseMatrix.zeros[Double](c._2.cols, c._2.rows))))
      .reduce((a,b) => a ++ b).toStream,
    num_rows, num_cols,
    num_row_blocks, num_col_blocks) {

  def _underlyingdata: Stream[((Long, Long), DenseMatrix[Double])] = underlyingdata

  override def t: UpperTriPartitionedMatrix =
    new UpperTriPartitionedMatrix(
      underlyingdata.map(c => (c._1.swap, c._2.t)),
      cols, rows, colBlocks, rowBlocks)

  override def repr: LowerTriPartitionedMatrix = this

  def \\[B, That](b: B)(implicit op: OpSolveMatrixBy.Impl2[LowerTriPartitionedMatrix, B, That]) =
    op.apply(repr, b)

}

/**
  * A partitioned upper triangular matrix.
  *
  * @author mandar2812 date: 18/10/2016
  * */
private[dynaml] class UpperTriPartitionedMatrix(
  underlyingdata: Stream[((Long, Long), DenseMatrix[Double])],
  num_rows: Long = -1L, num_cols: Long = -1L,
  num_row_blocks: Long = -1L, num_col_blocks: Long = -1L)
  extends PartitionedMatrix(
    data = underlyingdata
      .map(c =>
        if(c._1._1 == c._1._2) Seq(c)
        else Seq(c, (c._1.swap, DenseMatrix.zeros[Double](c._2.cols, c._2.rows))))
      .reduce((a,b) => a ++ b).toStream,
    num_rows, num_cols,
    num_row_blocks, num_col_blocks) {


  def _underlyingdata: Stream[((Long, Long), DenseMatrix[Double])] = underlyingdata

  override def t: LowerTriPartitionedMatrix =
    new LowerTriPartitionedMatrix(
      underlyingdata.map(c => (c._1.swap, c._2.t)),
      cols, rows, colBlocks, rowBlocks)

  override def repr: UpperTriPartitionedMatrix = this

  def \\[B, That](b: B)(implicit op: OpSolveMatrixBy.Impl2[UpperTriPartitionedMatrix, B, That]) =
    op.apply(repr, b)
}

/**
  * A partitioned positive semi-definite matrix.
  *
  * @author mandar2812 date: 18/10/2016
  * */
private[dynaml] class PartitionedPSDMatrix(
  underlyingdata: Stream[((Long, Long), DenseMatrix[Double])],
  num_rows: Long = -1L, num_cols: Long = -1L,
  num_row_blocks: Long = -1L, num_col_blocks: Long = -1L)
  extends PartitionedMatrix(
    data = underlyingdata
      .map(c =>
        if(c._1._1 == c._1._2) Seq(c)
        else Seq(c, (c._1.swap, c._2.t)))
      .reduce((a,b) => a ++ b).toStream,
    num_rows, num_cols,
    num_row_blocks, num_col_blocks) with
    NumericOps[PartitionedPSDMatrix] {

  override def t = this

  def _underlyingdata = underlyingdata

  override def repr: PartitionedPSDMatrix = this
}

object PartitionedPSDMatrix {

  /**
    * Create a [[PartitionedPSDMatrix]] instance from the outer product
    * of a [[PartitionedVector]] with itself.
    * */
  def fromOuterProduct(v: PartitionedVector): PartitionedPSDMatrix = {
    val mat: PartitionedMatrix = v*v.t

    val adjMat = (mat.L + mat.L.t).map(bm =>
      if(bm._1._1 == bm._1._2) (bm._1, bm._2*(DenseMatrix.eye[Double](bm._2.rows)*0.5))
      else bm)

    new PartitionedPSDMatrix(
      adjMat._data.filter(p => p._1._1 >= p._1._2),
      adjMat.rows, adjMat.cols, adjMat.rowBlocks, adjMat.colBlocks)
  }

}
