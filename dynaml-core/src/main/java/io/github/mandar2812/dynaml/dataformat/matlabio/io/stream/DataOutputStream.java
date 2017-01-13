package io.github.mandar2812.dynaml.dataformat.matlabio.io.stream;

import java.io.IOException;
import java.nio.ByteBuffer;

interface DataOutputStream 
{
    /**
     * Returns the current size of this stream.
     * 
     * @return the current size of this stream.
     * @throws IOException
     */
    public abstract int size() throws IOException;

    /**
     * Returns the current {@link ByteBuffer} mapped on the target file.
     * <p>
     * Note: the {@link ByteBuffer} has <strong>READ ONLY</strong> access.
     * 
     * @return the {@link ByteBuffer}
     * @throws IOException
     */
    public abstract ByteBuffer buffer() throws IOException;

    /**
     * Writes a sequence of bytes to this stream from the given buffer.
     * 
     * @param byteBuffer
     *            the source {@link ByteBuffer}
     * @throws IOException
     */
    public abstract void write(ByteBuffer byteBuffer) throws IOException;

}