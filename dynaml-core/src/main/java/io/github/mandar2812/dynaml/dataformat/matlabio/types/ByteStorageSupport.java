package io.github.mandar2812.dynaml.dataformat.matlabio.types;


public interface ByteStorageSupport<T extends Number>
{
    int getBytesAllocated();
    T buldFromBytes( byte[] bytes );
    byte[] getByteArray ( T value );
    Class<?> getStorageClazz();

}
