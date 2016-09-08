package io.github.mandar2812.dynaml.dataformat;
/*
 * Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 *
 *   * Neither the names of the Technical University of Berlin, ETH
 *   Zürich, or Fraunhofer FIRST nor the names of its contributors may
 *   be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

import java.util.regex.Pattern;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * <p>A class for reading and writing Arff-Files.</p>
 *
 * <p>You can either load a file, parse a string or a BufferedReader. Afterwards, you
 * can extract the information with the methods getComment(), getNumberOfAttributes(),
 * getAttributeName(), getAttributeType(), getAttributeData().</p>
 *
 * <p>Alternative, you can construct an empty ArffFile object, and then use setComment(),
 * defineAttribute(), add() to fill in the data and then save to a file with save(), or
 * to a string with write().</p>
 *
 * <p>The first comment in an ArffFile is extracted and made available through the *Comment()
 * accessors. Usually, this comment contains some interesting information about the data set.</p>
 *
 * <p>Currently, the class only supports numerical, string and nominal attributes. It also does
 * not support sparse storage (yet).</p>
 *
 * @author Mikio L. Braun, mikio@cs.tu-berlin.de
 */
public class ArffFile {

    private String relation;
    private String comment;
    private List<String> attribute_names;
    private Map<String, String> attribute_types;
    private Map<String, String[]> attribute_data;
    private List<Object[]> data;
    private int state;
    private final int COMMENT = 0;
    private final int HEADER = 1;
    private final int DATA = 2;
    private StringBuilder collectedComment;
    private int lineno;

    /** Load an ArffFile. */
    public static ArffFile load(String filename)
            throws FileNotFoundException, IOException, ArffFileParseError {
        ArffFile arff = new ArffFile();
        arff.parse(new BufferedReader(new FileReader(filename)));
        return arff;
    }

    /** Parse an ArffFile from a string. */
    public static ArffFile parse(String l)
            throws IOException, ArffFileParseError {
        ArffFile arff = new ArffFile();
        arff.parse(new BufferedReader(new StringReader(l)));
        return arff;
    }

    /** Construct an empty ArffFile. */
    public ArffFile() {
        relation = "";
        comment = null;
        attribute_names = new ArrayList<String>();
        attribute_types = new HashMap<String, String>();
        attribute_data = new HashMap<String, String[]>();
        data = new ArrayList<Object[]>();
    }

    /** Parse an ArffFile from a BufferedReader. */
    public void parse(BufferedReader r) throws IOException {
        initParsing();
        String line;
        lineno = 1;
        boolean hadErrors = false;
        while ((line = r.readLine()) != null) {
            try {
                parseLine(line);
            } catch (ArffFileParseError e) {
                hadErrors = true;
                System.err.println(e.getMessage());
            }
            lineno++;
        }
    }

    private void initParsing() {
        state = COMMENT;
        collectedComment = new StringBuilder();
    }

    private void parseLine(String line) throws ArffFileParseError {
        switch (state) {
            case COMMENT:
                if (!line.isEmpty() && line.charAt(0) == '%') {
                    if (line.length() >= 2)
                        collectedComment.append(line.substring(2));
                    appendNewline(collectedComment);
                } else {
                    comment = collectedComment.toString();
                    state = HEADER;
                    parseLine(line);
                }
                break;
            case HEADER:
                String lowerline = line.toLowerCase();
                if (lowerline.startsWith("@relation")) {
                    parseRelationDefinition(line);
                } else if (lowerline.startsWith("@attribute")) {
                    try {
                        parseAttributeDefinition(line);
                    } catch (ArffFileParseError e) {
                        System.err.println("Warning: " + e.getMessage());
                    }
                } else if (lowerline.startsWith("@data")) {
                    state = DATA;
                }
                break;

            case DATA:
                if (!line.isEmpty() && line.charAt(0) != '%')
                    parseData(line);
                break;
        }
    }

    private void parseRelationDefinition(String line) {
        int i = line.indexOf(' ');
        relation = line.substring(i + 1);
    }

    /** Define a new attribute. Type must be one of "numeric", "string", and
     * "nominal". For nominal attributes, the allowed values
     * must also be given. This variant of defineAttribute allows to set this data.
     */
    public void defineAttribute(String name, String type, String[] data) {
        attribute_names.add(name);
        attribute_types.put(name, type);
        if (data != null)
            attribute_data.put(name, data);
    }

    private void parseAttributeDefinition(String line) throws ArffFileParseError {
        Scanner s = new Scanner(line);
        Pattern p = Pattern.compile("[a-zA-Z_][a-zA-Z0-9_]*|\\{[^\\}]+\\}|\\'[^\\']+\\'|\\\"[^\\\"]+\\\"");
        String keyword = s.findInLine(p);
        String name = s.findInLine(p);
        String type = s.findInLine(p);

        if (name == null || type == null) {
            throw new ArffFileParseError(lineno, "Attribute definition cannot be parsed");
        }

        String lowertype = type.toLowerCase();

        if (lowertype.equals("real") || lowertype.equals("numeric") || lowertype.equals("integer")) {
            defineAttribute(name, "numeric", null);
        } else if (lowertype.equals("string")) {
            defineAttribute(name, "string", null);
        } else if (type.startsWith("{") && type.endsWith("}")) {
            type = type.substring(1, type.length() - 1);
            type = type.trim();
            defineAttribute(name, "nominal", type.split("\\s*,\\s*"));
        } else {
            throw new ArffFileParseError(lineno, "Attribute of type \"" + type + "\" not supported (yet)");
        }
    }

    private void parseData(String line) throws ArffFileParseError {
        int num_attributes = attribute_names.size();
        if (line.charAt(0) == '{' && line.charAt(line.length() - 1) == '}') {
            throw new ArffFileParseError(lineno, "Sparse data not supported (yet).");
        } else {
            String[] tokens = line.split("\\s*,\\s*");
            if (tokens.length != num_attributes) {
                throw new ArffFileParseError(lineno, "Warning: line " + lineno + " does not contain the right " +
                        "number of elements (should be " + num_attributes + ", got " + tokens.length + ".");
            }

            Object[] datum = new Object[num_attributes];
            for (int i = 0; i < num_attributes; i++) {
                //System.out.printf("line %d token %d: %s%n", lineno, i, tokens[i]);
                String name = attribute_names.get(i);
                String at = attribute_types.get(name);
                if (at.equals("numeric")) {
                    datum[i] = Double.parseDouble(tokens[i]);
                } else if (at.equals("string")) {
                    datum[i] = tokens[i];
                } else if (at.equals("nominal")) {
                    if (!isNominalValueValid(name, tokens[i])) {
                        throw new ArffFileParseError(lineno, "Undefined nominal value \"" +
                                tokens[i] + "\" for field " + name + ".");
                    }
                    datum[i] = tokens[i];
                }
            }

            data.add(datum);
        }
    }

    private boolean isNominalValueValid(String name, String token) throws ArffFileParseError {
        String[] values = attribute_data.get(name);
        boolean found = false;
        for (int t = 0; t < values.length; t++) {
            if (values[t].equals(token)) {
                found = true;
            }
        }
        return found;
    }

    /** Generate a string which describes the data set. */
    public String dump() {
        StringBuilder s = new StringBuilder();

        s.append("Relation " + relation); appendNewline(s);
        s.append("with attributes"); appendNewline(s);
        for (String n : attribute_names) {
            s.append("   " + n + " of type " + attribute_types.get(n));
            if (attribute_types.get(n).equals("nominal")) {
                s.append(" with values ");
                joinWith(attribute_data.get(n), s, ", ");
            }
            appendNewline(s);
        }

        appendNewline(s);
        s.append("Data (first 10 lines of " + data.size() + "):"); appendNewline(s);

        for (int i = 0; i < Math.min(data.size(), 10); i++) {
            Object[] datum = data.get(i);
            joinWith(datum, s, ", ");
            appendNewline(s);
        }
        return s.toString();
    }

    /** Formats an array of Objects in the passed StringBuilder using toString()
     * and using del as the delimiter.
     *
     * For example, on <tt>objects = { 1, 2, 3 }</tt>, and <tt>del = " + "</tt>, you get
     * <tt>"1 + 2 + 3"</tt>.
     */
    private void joinWith(Object[] objects, StringBuilder s, String del) {
        boolean first = true;
        for (Object o : objects) {
            if (!first) {
                s.append(del);
            }
            s.append(o.toString());
            first = false;
        }
    }

    /** Write the ArffFile to a string. */
    public String write() {
        StringBuilder s = new StringBuilder();

        if (comment != null) {
            s.append("% ");
            s.append(comment.replaceAll(System.getProperty("line.separator"), System.getProperty("line.separator") + "% "));
            appendNewline(s);

            s.append("@relation " + relation); appendNewline(s);

            for (String name : attribute_names) {
                s.append("@attribute ");
                s.append(name);
                s.append(" ");

                String type = attribute_types.get(name);
                if (type.equals("numeric")) {
                    s.append("numeric");
                }
                else if (type.equals("string"))
                    s.append("string");
                else if (type.equals("nominal")) {
                    s.append("{");
                    joinWith(attribute_data.get(name), s, ",");
                    s.append("}");
                }
                appendNewline(s);
            }

            s.append("@data");
            appendNewline(s);

            for (Object[] datum : data) {
                joinWith(datum, s, ",");
                appendNewline(s);
            }
        }

        return s.toString();
    }

    /** Save the data into a file. */
    public void save(String filename) throws IOException {
        Writer w = new FileWriter(filename);
        w.write(write());
    }

    private void appendNewline(StringBuilder s) {
        s.append(System.getProperty("line.separator"));
    }

    /** Main function for debugging. Loads files from the argument list
     * and dumps their contents to the System.out.
     */
    public static void main(String[] args) {
        String file = "% oh yes, this is great!\n" +
                "% and even better than I thought!\n" +
                "@relation foobar\n" +
                "@attribute number real\n" +
                "@attribute argh {yes, no}\n" +
                "@data\n" +
                "1, yes\n" +
                "0, no\n";

        try {
            //ArffFile arff = parse(file);
            //System.out.println(arff.dump());
            //System.out.println(arff.write());

            for (String filename : args) {
                ArffFile arff = load(filename);
                System.out.println(arff.dump());
            }
        } catch (ArffFileParseError e) {
            System.out.println("Couldn't parse file!");
        } catch (IOException e) {
            System.out.println("Ouch");
        }
    }

    /**
     * Get the name of the relation.
     */
    public String getRelation() {
        return relation;
    }

    /**
     * Set the name of the relation.
     */
    public void setRelation(String relation) {
        this.relation = relation;
    }

    /**
     * Get the initial comment of the relation.
     */
    public String getComment() {
        return comment;
    }

    /**
     * Set the initial comment of the relation.
     */
    public void setComment(String comment) {
        this.comment = comment;
    }

    /**
     * Get the number of attributes.
     */
    public int getNumberOfAttributes() {
        return attribute_names.size();
    }

    /**
     * Get the name of an attribute.
     */
    public String getAttributeName(int idx) {
        return attribute_names.get(idx);
    }

    /**
     * Get the type of an attribute. Currently, the attribute types are
     * "numeric", "string", and "nominal". For nominal attributes, use getAttributeData()
     * to retrieve the possible values for the attribute.
     */
    public String getAttributeType(String name) {
        return attribute_types.get(name);
    }

    /**
     * Get additional information on the attribute. This data is used for
     * nominal attributes to define the possible values.
     */
    public String[] getAttributeData(String name) {
        return attribute_data.get(name);
    }

    /** Add a datum. No checking of the data types is performed! */
    public void add(Object[] datum) {
        data.add(datum);
    }

    public int getDataSize() {
        return data.size();
    }

    public Object[] getDatum(int i) {
        return data.get(i);
    }

    public List<Object[]> getData() {
        return data;
    }
}
