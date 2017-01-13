package io.github.mandar2812.dynaml.dataformat.matlabio.types;

public class MLObject extends MLArray
{
    private final MLStructure o;
    private final String className;
    
    public MLObject( String name, String className, MLStructure o )
    {
        super( name, new int[] {1, 1}, MLArray.mxOBJECT_CLASS, 0 );
        this.o = o;
        this.className = className;
    }

    public String getClassName()
    {
        return className;
    }
    
    public MLStructure getObject()
    {
        return o;
    }
}
