!!! summary "Partitioned Vectors & Dual Vectors"
    The `#!scala dynaml.algebra` package contains a number of classes and utilities
    for constructing blocked linear algebra objects.


DynaML makes extensive use of the [breeze](https://github.com/scalanlp/breeze) linear algebra library for matrix-vector
operations. Breeze is an attractive option because it is easy to use and has the ability to use low level implementations
such as _LAPACK_ and _BLAS_ for performance speed-up.

If we are working with linear algebra objects which are large in size as compared to the available JVM memory, it may be necessary
to not construct the entire vector in an eager fashion.

In the Scala language, there are _lazy_ data structures, which are not computed unless they are required in further
computation. The `dynaml.algebra` package leverages lazy data structures to create blocked vectors and matrices.
Each partition/block of a blocked object is a breeze vector or matrix.

The proceeding pages give the user a glimpse of how to use and manipulate objects of the `algebra` package.
