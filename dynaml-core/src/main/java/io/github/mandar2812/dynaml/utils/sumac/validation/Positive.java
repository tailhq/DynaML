package io.github.mandar2812.dynaml.utils.sumac.validation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Use this annotation on any arg-field which must have a positive (> 0) value to add an automatic validation function.
 *
 * It is meaningless if applied to a non-numeric field.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Positive {}
