package io.github.mandar2812.dynaml.dataformat.matlabio.io.stream;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;

public abstract class BufferedOutputStream extends OutputStream
{
    /**
     * Returns the backing {@link ByteBuffer}
     * @return
     */
    public abstract ByteBuffer buffer() throws IOException;
    public abstract long       size() throws IOException;
}
