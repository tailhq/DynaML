/**
 * 
 */
package io.github.mandar2812.dynaml.dataformat.matlabio.io.stream;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLArray;

/**
 * This is an {@link OutputStream} that is backed by a {@link RandomAccessFile}
 * and accessed with buffered access.
 * 
 * @author Wojciech Gradkowski (<a
 *         href="mailto:wgradkowski@gmail.com">wgradkowski@gmail.com</a>)
 * 
 */
public class FileBufferedOutputStream extends BufferedOutputStream
{
    private static final int BUFFER_SIZE = 1024;
    private ByteBuffer buf;
    private FileChannel rwChannel;
    private RandomAccessFile raFile;
    private final File file;
    
    public FileBufferedOutputStream() throws IOException
    {
        file = File.createTempFile( "jmatio-", null );
        file.deleteOnExit();
        raFile = new RandomAccessFile(file, "rw");
        rwChannel = raFile.getChannel();
        buf = ByteBuffer.allocate( BUFFER_SIZE );
    }
    
    public FileBufferedOutputStream( MLArray array ) throws IOException
    {
        file = File.createTempFile( "jmatio-" + array.getName() + "-", null );
        file.deleteOnExit();
        raFile = new RandomAccessFile(file, "rw");
        rwChannel = raFile.getChannel();
        buf = ByteBuffer.allocate( BUFFER_SIZE );
    }
    
    @Override
    public void write(int b) throws IOException
    {
        if ( buf.position() >= buf.capacity() )
        {
            flush();
        }
        
        buf.put( (byte) (b & 0xff) );
    }
    
    /* (non-Javadoc)
     * @see java.io.OutputStream#write(byte[])
     */
    @Override
    public void write(byte[] b) throws IOException
    {
        write(b, 0, b.length);
    }

    /* (non-Javadoc)
     * @see java.io.OutputStream#write(byte[], int, int)
     */
    @Override
    public void write(byte[] b, int off, int len) throws IOException
    {
        int wbytes = len;
        int offset = off;
        
        while( wbytes > 0 )
        {
            if ( buf.position() >= buf.capacity() )
            {
                flush();
            }
            
            int length = Math.min( wbytes, buf.limit() - buf.position() );
            
            buf.put(b, offset, length);
            
            offset += length;
            wbytes -= length;
        }
    }

    /* (non-Javadoc)
     * @see java.io.OutputStream#close()
     */
    @Override
    public void close() throws IOException
    {
        flush();
        
        buf = null;
        
        if ( rwChannel.isOpen() )
        {

            rwChannel.close();
        }
        
        raFile.close();
        rwChannel = null;
        raFile = null;
    }

    /* (non-Javadoc)
     * @see java.io.OutputStream#flush()
     */
    @Override
    public void flush() throws IOException
    {
        if ( buf != null && buf.position() > 0 )
        {    
            buf.flip();
            rwChannel.write( buf );
            buf.clear();
        }
    }

    /* (non-Javadoc)
     * @see io.github.mandar2812.dynaml.dataformat.matlabio.io.DataOutputStream#size()
     */
    public long size() throws IOException
    {
        flush();
        
        return (int) file.length();
    }

    /* (non-Javadoc)
     * @see io.github.mandar2812.dynaml.dataformat.matlabio.io.DataOutputStream#getByteBuffer()
     */
    @Override
    public ByteBuffer buffer() throws IOException
    {
        return rwChannel.map( FileChannel.MapMode.READ_ONLY, 0, file.length() );
    }

    /* (non-Javadoc)
     * @see io.github.mandar2812.dynaml.dataformat.matlabio.io.DataOutputStream#write(java.nio.ByteBuffer)
     */
    public void write( ByteBuffer byteBuffer ) throws IOException
    {
        byte[] tmp = new byte[BUFFER_SIZE]; 
        
        while ( byteBuffer.hasRemaining() )
        {
            int length = Math.min( byteBuffer.remaining(), tmp.length );
            byteBuffer.get( tmp, 0, length);
            write(tmp, 0, length);
        }
    }
 
}