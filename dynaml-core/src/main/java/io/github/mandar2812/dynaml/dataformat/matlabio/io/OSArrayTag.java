package io.github.mandar2812.dynaml.dataformat.matlabio.io;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Tiny class that represents MAT-file TAG 
 * It simplifies writing data. Automates writing padding for instance.
 */
class OSArrayTag extends MatTag
{
    private ByteBuffer data;

    /**
     * Creates TAG and stets its <code>size</code> as size of byte array
     * 
     * @param type
     * @param data
     */
    public OSArrayTag(int type, byte[] data )
    {
        this ( type, ByteBuffer.wrap( data ) );
    }
    /**
     * Creates TAG and stets its <code>size</code> as size of byte array
     * 
     * @param type
     * @param data
     */
    public OSArrayTag(int type, ByteBuffer data )
    {
        super( type, data.limit() );
        this.data = data;
        data.rewind();
    }

    
    /**
     * Writes tag and data to <code>DataOutputStream</code>. Wites padding if neccesary.
     * 
     * @param os
     * @throws IOException
     */
    public void writeTo(DataOutputStream os) throws IOException
    {
    
    	int padding;
		if (size<=4 && size>0) {
			// Use small data element format (Page 1-10 in "MATLAB 7 MAT-File Format", September 2010 revision)
    		os.writeShort(size);
    		os.writeShort(type);
            padding = getPadding(data.limit(), true);
    	} else {
    		os.writeInt(type);
    		os.writeInt(size);
            padding = getPadding(data.limit(), false);
    	}
        
        int maxBuffSize = 1024;
        int writeBuffSize = data.remaining() < maxBuffSize ? data.remaining() : maxBuffSize;
        byte[] tmp = new byte[writeBuffSize]; 
        while ( data.remaining() > 0 )
        {
            int length = data.remaining() > tmp.length ? tmp.length : data.remaining();
            data.get( tmp, 0, length);
            os.write(tmp, 0, length);
        }
        
        if ( padding > 0 )
        {
            os.write( new byte[padding] );
        }
    }
}