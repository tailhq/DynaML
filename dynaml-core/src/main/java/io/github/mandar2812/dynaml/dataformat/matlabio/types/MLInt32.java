package io.github.mandar2812.dynaml.dataformat.matlabio.types;

import java.nio.ByteBuffer;


public class MLInt32 extends MLNumericArray<Integer>
{

    /**
     * Normally this constructor is used only by MatFileReader and MatFileWriter
     * 
     * @param name - array name
     * @param dims - array dimensions
     * @param type - array type: here <code>mxDOUBLE_CLASS</code>
     * @param attributes - array flags
     */
    public MLInt32( String name, int[] dims, int type, int attributes )
    {
        super( name, dims, type, attributes );
    }
    /**
     * Create a <code>{@link MLInt64}</code> array with given name,
     * and dimensions.
     * 
     * @param name - array name
     * @param dims - array dimensions
     */
    public MLInt32(String name, int[] dims)
    {
        super(name, dims, MLArray.mxINT32_CLASS, 0);
    }
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a 2D real matrix from a one-dimensional packed array
     * 
     * @param name - array name
     * @param vals - One-dimensional array of doubles, packed by columns (ala Fortran).
     * @param m - Number of rows
     */
    public MLInt32(String name, Integer[] vals, int m )
    {
        super(name, MLArray.mxINT32_CLASS, vals, m );
    }
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a 2D real matrix from <code>byte[][]</code>
     * 
     * Note: array is converted to Byte[]
     * 
     * @param name - array name
     * @param vals - two-dimensional array of values
     */
    public MLInt32( String name, int[][] vals )
    {
        this( name, int2DToInteger(vals), vals.length );
    }
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a matrix from a one-dimensional packed array
     * 
     * @param name - array name
     * @param vals - One-dimensional array of doubles, packed by columns (ala Fortran).
     * @param m - Number of rows
     */
    public MLInt32(String name, int[] vals, int m)
    {
        this(name, castToInteger( vals ), m );
    }
    /**
     * Gets two-dimensional real array.
     * 
     * @return - 2D real array
     */
    public int[][] getArray()
    {
        int[][] result = new int[getM()][];
        
        for ( int m = 0; m < getM(); m++ )
        {
           result[m] = new int[ getN() ];

           for ( int n = 0; n < getN(); n++ )
           {               
               result[m][n] = getReal(m,n);
           }
        }
        return result;
    }
    /**
     * Casts <code>Double[]</code> to <code>byte[]</code>
     * 
     * @param - source <code>Long[]</code>
     * @return - result <code>long[]</code>
     */
    private static Integer[] castToInteger( int[] d )
    {
        Integer[] dest = new Integer[d.length];
        for ( int i = 0; i < d.length; i++ )
        {
            dest[i] = d[i];
        }
        return dest;
    }
    /**
     * Converts byte[][] to Long[]
     * 
     * @param dd
     * @return
     */
    private static Integer[] int2DToInteger( int[][] dd )
    {
        Integer[] d = new Integer[ dd.length*dd[0].length ];
        for ( int n = 0; n < dd[0].length; n++ )
        {
            for ( int m = 0; m < dd.length; m++ )
            {
                d[ m+n*dd.length ] = dd[m][n]; 
            }
        }
        return d;
    }
    public Integer buldFromBytes(byte[] bytes)
    {
        if ( bytes.length != getBytesAllocated() )
        {
            throw new IllegalArgumentException( 
                        "To build from byte array I need array of size: " 
                                + getBytesAllocated() );
        }
        return ByteBuffer.wrap( bytes ).getInt();
    }
    public int getBytesAllocated()
    {
        return Integer.SIZE >> 3;
    }
    
    public Class<Integer> getStorageClazz()
    {
        return Integer.class;
    }
    public byte[] getByteArray(Integer value)
    {
        int byteAllocated = getBytesAllocated();
        ByteBuffer buff = ByteBuffer.allocate( byteAllocated );
        buff.putInt( value );
        return buff.array();
    }

}
