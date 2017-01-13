package io.github.mandar2812.dynaml.dataformat.matlabio.types;

public class MLJavaObject extends MLArray
{
    private final Object o;
    private final String className;
    
    public MLJavaObject( String name, String className, Object o )
    {
        super( name, new int[] {1, 1}, MLArray.mxOPAQUE_CLASS, 0 );
        this.o = o;
        this.className = className;
    }

    public String getClassName()
    {
        return className;
    }
    
    public Object getObject()
    {
        return o;
    }

}
