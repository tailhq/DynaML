package io.github.mandar2812.dynaml.dataformat.matlabio.io.stream;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * From https://gist.github.com/manzke/985007
 *
 */
public class ByteBufferedOutputStream extends BufferedOutputStream
{
    private ByteBuffer buffer;

    private boolean onHeap;

    private float increasing = DEFAULT_INCREASING_FACTOR;

    public static final float DEFAULT_INCREASING_FACTOR = 1.5f;

    public ByteBufferedOutputStream( int size )
    {
        this( size, DEFAULT_INCREASING_FACTOR, false );
    }

    public ByteBufferedOutputStream( int size, boolean onHeap )
    {
        this( size, DEFAULT_INCREASING_FACTOR, onHeap );
    }

    public ByteBufferedOutputStream( int size, float increasingBy )
    {
        this( size, increasingBy, false );
    }

    public ByteBufferedOutputStream( int size, float increasingBy, boolean onHeap )
    {
        if ( increasingBy <= 1 )
        {
            throw new IllegalArgumentException( "Increasing Factor must be greater than 1.0" );
        }
        if ( onHeap )
        {
            buffer = ByteBuffer.allocate( size );
        }
        else
        {
            buffer = ByteBuffer.allocateDirect( size );
        }
        this.onHeap = onHeap;
    }

    @Override
    public void write( byte[] b, int off, int len ) throws IOException
    {
        int position = buffer.position();
        int limit = buffer.limit();

        long newTotal = position + len;
        if ( newTotal > limit )
        {
            int capacity = (int) ( buffer.capacity() * increasing );
            while ( capacity <= newTotal )
            {
                capacity = (int) ( capacity * increasing );
            }

            increase( capacity );
        }

        buffer.put( b, 0, len );
    }

    @Override
    public void write( int b ) throws IOException
    {
        if ( !buffer.hasRemaining() )
        {
            increase( (int) ( buffer.capacity() * increasing ) );
        }
        buffer.put( (byte) b );
    }

    protected void increase( int newCapacity )
    {
        buffer.limit( buffer.position() );
        buffer.rewind();

        ByteBuffer newBuffer;
        if ( onHeap )
        {
            newBuffer = ByteBuffer.allocate( newCapacity );
        }
        else
        {
            newBuffer = ByteBuffer.allocateDirect( newCapacity );
        }

        newBuffer.put( buffer );
        buffer.clear();
        buffer = newBuffer;
    }
    
    @Override
    public long size()
    {
        return buffer.position();
    }

    public long capacity()
    {
        return buffer.capacity();
    }

    public ByteBuffer buffer()
    {
        return buffer;
    }
}
