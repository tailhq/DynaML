package io.github.mandar2812.dynaml.utils.sumac.validation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Use this annotation on any arg-field which must be given a value distinct from its "default" value.  It will
 * automatically add a validation function.
 *
 * Usually, the "default" value of a field is whatever the value is with a no-arg constructor.  Note that this
 * annotation does *not* just check that some argument was given.  The argument must be distinct from the default
 * value.  This means that you should make the default value of the field something that is *invalid*.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Required {}
