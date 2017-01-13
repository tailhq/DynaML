package io.github.mandar2812.dynaml.dataformat.matlabio.common.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLArray;
import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLCell;
import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLChar;
import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLNumericArray;
import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLObject;
import io.github.mandar2812.dynaml.dataformat.matlabio.types.MLStructure;

/**
 * The JMatIO query parser. Allows to use Matlab-like syntax to access {@link MLArray} objects. 
 * <p>
 * 
 * @author wgradkowski
 *
 */
public class MLArrayQuery
{
    private String queryString;
    
    private static final String regexp = "([a-zA-Z0-9]+)(\\(([0-9]+|:)(,([0-9:]+|:))?\\))?\\.?";
    private static final Pattern pat = Pattern.compile( regexp );
    
    public MLArrayQuery( String queryString )
    {
        if ( !Pattern.matches( "^(" + regexp + ")+$", queryString ) )
        {
            throw new IllegalArgumentException();
        }
        
        this.queryString = queryString;
    }
    
    /**
     * 
     * 
     * @param array
     * @param query
     * @return
     */
    public static Object q( MLArray array, String query )
    {
        MLArrayQuery q = new MLArrayQuery( query );
        
        return q.query( array );
    }
    
    /**
     * Parses the query string and returns the object it refers to.
     * 
     * @param array
     *            source {@link MLArray}
     * @return query result
     */
    public Object query( MLArray array )
    {
        Matcher mat = pat.matcher( queryString );
        
        MLArray current = null;
        
        int prevM = 0;
        int prevN = 0;
        
        while ( mat.find() )
        {
            String name   = mat.group( 1 );
            String rangeM = mat.group( 3 );
            String rangeN = mat.group( 5 );
            
            int m = rangeM != null ? Integer.parseInt( rangeM ) -1 : -1;
            int n = rangeN != null ? Integer.parseInt( rangeN ) -1 : -1;
            
            if ( current == null )
            {
                current = array;
                
                if ( !current.getName().equals( name ) && !current.getName().equals( "@" ) )
                {
                    throw new RuntimeException("No such array or field <" + name + "> in <" + (current != null ? current.getName() : "/") + ">" );
                }
                
                prevM = m;
                prevN = n;
                
                continue;
            }
            
            int type = current.getType();
            
            switch ( type )
            {
                case MLArray.mxOBJECT_CLASS:
                    {
                        MLObject object = (MLObject) current;
                    
                        MLArray field  = object.getObject().getField( name, prevM, prevN );
                        
                        if ( field == null )
                        {
                            throw new RuntimeException("no such field: " + name );
                        }
                        current = field;
                    }
                    break;
                case MLArray.mxSTRUCT_CLASS:
                    {
                        MLStructure struct = (MLStructure) current;
                        
                        MLArray field  = struct.getField( name, prevM > 0 ? prevM : 0, prevN > 0 ? prevN : 0 );
                        
                        if ( field == null )
                        {
                            throw new RuntimeException("no such field: " + name );
                        }
                        current = field;
                    }
                    break;
                case MLArray.mxCELL_CLASS:
                    {
                        MLCell mlcell = (MLCell) current;
                        if ( m > -1 && n > -1 )
                        {
                            current = mlcell.get( m, n );
                        }
                        else if ( m > -1 )
                        {
                            current = mlcell.get( m );
                        }
                        else
                        {
                            throw new RuntimeException();
                        }
                    }
                    break;
                default:
            }
            
            prevM = m;
            prevN = n;
        }
        
        return getContent(current, prevM, prevN );
    }

    /**
     * Returns the content of the field/cell/object.
     * 
     * @param array
     *            the parent structure/cell
     * @param m
     *            column or -1
     * @param n
     *            row or -1
     * @return if both m and n are -1, returns {@link MLArray}, if n is -1, returns
     *         content under index m, if both m and n are not-negative, returns
     *         content of (m,n)
     */
    public Object getContent( MLArray array, int m, int n )
    {
        int type = array.getType();
        
        Object result = null;
        
        switch ( type )
        {
            case MLArray.mxINT8_CLASS:
            case MLArray.mxINT16_CLASS:
            case MLArray.mxINT32_CLASS:
            case MLArray.mxINT64_CLASS:
            case MLArray.mxUINT8_CLASS:
            case MLArray.mxUINT16_CLASS:
            case MLArray.mxUINT32_CLASS:
            case MLArray.mxUINT64_CLASS:
            case MLArray.mxSINGLE_CLASS:
            case MLArray.mxDOUBLE_CLASS:
                MLNumericArray<?> numeric = (MLNumericArray<?>) array;
                if ( m > -1 && n > -1 )
                {
                    result = numeric.get( m, n );
                }
                else if ( m > -1 )
                {
                    result = numeric.get( m );
                }
                else
                {
                    result = array;
                }
                break;
            case MLArray.mxCHAR_CLASS:
                MLChar mlchar = (MLChar) array;
                if ( m > -1 && n > -1 )
                {
                    result = mlchar.getChar( m, n );
                }
                else if ( m > -1 )
                {
                    result = mlchar.getString( m );
                }
                else
                {
                    result = mlchar;
                }
                break;
            case MLArray.mxCELL_CLASS:
                MLCell mlcell = (MLCell) array;
                if ( m > -1 && n > -1 )
                {
                    result = getContent( mlcell.get( m, n ), 0, -1);
                }
                else if ( m > -1 )
                {
                    result = getContent( mlcell.get( m ), 0, -1 );
                }
                else
                {
                    result = getContent( mlcell.get( 0 ), -1, -1 );
                }
                break;
            default:
                result = array;
        }
        
        return result;
    }
    

    
}
