package io.github.mandar2812.dynaml.utils.annotation;

import java.lang.annotation.*;

/**
 * Experimental classes/objects/functions which may be removed later.
 *
 * */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.FIELD, ElementType.METHOD, ElementType.PARAMETER,
        ElementType.CONSTRUCTOR, ElementType.LOCAL_VARIABLE, ElementType.PACKAGE})
public @interface Experimental {}
