package io.github.mandar2812.dynaml.dataformat.matlabio.types;

public class MLUInt32 extends MLInt32
{

    public MLUInt32( String name, int[] dims, int type, int attributes )
    {
        super( name, dims, type, attributes );
    }

    public MLUInt32( String name, int[] vals, int m )
    {
        super( name, vals, m );
    }

    public MLUInt32( String name, int[] dims )
    {
        super( name, dims );
    }

    public MLUInt32( String name, int[][] vals )
    {
        super( name, vals );
    }

    public MLUInt32( String name, Integer[] vals, int m )
    {
        super( name, vals, m );
    }
}
