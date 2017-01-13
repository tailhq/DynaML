package io.github.mandar2812.dynaml.dataformat.matlabio.types;

import java.nio.ByteBuffer;

/**
 * Class represents Int16 array (matrix)
 * 
 */
public class MLInt16 extends MLNumericArray<Short>
{

    /**
     * Normally this constructor is used only by MatFileReader and MatFileWriter
     * 
     * @param name - array name
     * @param dims - array dimensions
     * @param type - array type: here <code>mxINT16_CLASS</code>
     * @param attributes - array flags
     */
    public MLInt16( String name, int[] dims, int type, int attributes )
    {
        super( name, dims, type, attributes );
    }
    
    /**
     * Create a <code>MLSingle</code> array with given name,
     * and dimensions.
     * 
     * @param name - array name
     * @param dims - array dimensions
     */
    public MLInt16(String name, int[] dims)
    {
        super(name, dims, MLArray.mxINT16_CLASS, 0);
    }
    
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a 2D real matrix from a one-dimensional packed array
     * 
     * @param name - array name
     * @param vals - One-dimensional array of Short, packed by columns (ala Fortran).
     * @param m - Number of rows
     */
    public MLInt16(String name, Short[] vals, int m )
    {
        super(name, MLArray.mxINT16_CLASS, vals, m );
    }
    
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a 2D real matrix from <code>double[][]</code>
     * 
     * Note: array is converted to Short[]
     * 
     * @param name - array name
     * @param vals - two-dimensional array of values
     */
    public MLInt16( String name, short[][] vals )
    {
        this( name, short2DToShort(vals), vals.length );
    }
    
    /**
     * <a href="http://math.nist.gov/javanumerics/jama/">Jama</a> [math.nist.gov] style: 
     * construct a matrix from a one-dimensional packed array
     * 
     * @param name - array name
     * @param vals - One-dimensional array of short, packed by columns (ala Fortran).
     * @param m - Number of rows
     */
    public MLInt16(String name, short[] vals, int m)
    {
        this(name, castToShort( vals ), m );
    }
    /**
     * Gets two-dimensional real array.
     * 
     * @return - 2D real array
     */
    public short[][] getArray()
    {
        short[][] result = new short[getM()][];
        
        for ( int m = 0; m < getM(); m++ )
        {
           result[m] = new short[ getN() ];

           for ( int n = 0; n < getN(); n++ )
           {               
               result[m][n] = getReal(m,n);
           }
        }
        return result;
    }
    
    /**
     * Casts <code>Short[]</code> to <code>short[]</code>
     * 
     * @param - source <code>short[]</code>
     * @return - result <code>Short[]</code>
     */
    private static Short[] castToShort( short[] d )
    {
        Short[] dest = new Short[d.length];
        for ( int i = 0; i < d.length; i++ )
        {
            dest[i] = (Short)d[i];
        }
        return dest;
    }
    
    /**
     * Converts short[][] to Short[]
     * 
     * @param dd
     * @return
     */
    private static Short[] short2DToShort( short[][] dd )
    {
        Short[] d = new Short[ dd.length*dd[0].length ];
        for ( int n = 0; n < dd[0].length; n++ )
        {
            for ( int m = 0; m < dd.length; m++ )
            {
                d[ m+n*dd.length ] = dd[m][n]; 
            }
        }
        return d;
    }
    
    public int getBytesAllocated()
    {
        return Short.SIZE >> 3;
    }
    
    public Short buldFromBytes(byte[] bytes)
    {
        if ( bytes.length != getBytesAllocated() )
        {
            throw new IllegalArgumentException( 
                        "To build from byte array I need array of size: " 
                                + getBytesAllocated() );
        }
        return ByteBuffer.wrap( bytes ).getShort();
        
    }
    
    public byte[] getByteArray(Short value)
    {
        int byteAllocated = getBytesAllocated();
        ByteBuffer buff = ByteBuffer.allocate( byteAllocated );
        buff.putShort( value );
        return buff.array();
    }
    
    public Class<Short> getStorageClazz()
    {
        return Short.class;
    }
}
