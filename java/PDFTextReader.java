
import org.apache.pdfbox.contentstream.operator.color.*;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.text.LegacyPDFStreamEngine;
import org.apache.pdfbox.text.TextPosition;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class PDFTextReader extends LegacyPDFStreamEngine {
    final List<Float> columns;
    Map<Integer, List<Pair<String, XY>>> dataSet = new HashMap<>();
    final Map<String, List<TextAndColor>> textItems = new TreeMap<>();

    public PDFTextReader(List<Float> colsAt) throws IOException {
        this.columns = colsAt;
        addOperator(new SetStrokingColorSpace());
        addOperator(new SetNonStrokingColorSpace());
        addOperator(new SetStrokingDeviceCMYKColor());
        addOperator(new SetNonStrokingDeviceCMYKColor());
        addOperator(new SetNonStrokingDeviceRGBColor());
        addOperator(new SetStrokingDeviceRGBColor());
        addOperator(new SetNonStrokingDeviceGrayColor());
        addOperator(new SetStrokingDeviceGrayColor());
        addOperator(new SetStrokingColor());
        addOperator(new SetStrokingColorN());
        addOperator(new SetNonStrokingColor());
        addOperator(new SetNonStrokingColorN());
    }

    private static class TextAndColor {
        public final TextPosition tp;
        public final int color;

        public TextAndColor(TextPosition tp, int color) {
            this.tp = tp;
            this.color = color;
        }

        public Float getX() {
            return tp.getX();
        }

        public Float getY() {
            return tp.getY();
        }

        public Float getHeight() {
            return tp.getHeight();
        }

        public Float getFontSize() {
            return tp.getFontSize();
        }

        public PDFont getFont() {
            return tp.getFont();
        }

        public String getUnicode() {
            return tp.getUnicode();
        }
    }

    public static class StringFontColor {
        public String text;
        public Float fontSize;
        public String fontName;
        public Integer color;
        public Float x;
        public Float y;
        public Float h;

        public StringFontColor(String text, Float fontSize, String fontName, Integer color, Float x, Float y, Float h) {
            this.text = (text == null) ? "" : text;
            this.fontSize = (fontSize == null) ? 0.0f : fontSize;
            this.fontName = (fontName == null) ? "" : fontName;
            this.color = (color == null) ? 0 : color;
            this.x = (x == null) ? 0.0f : x;
            this.y = (y == null) ? 0.0f : y;
            this.h = (h == null) ? 0.0f : h;
        }
    }

    protected void processTextPosition(TextPosition t) {
        super.processTextPosition(t);
        int color = 0;
        try {
            color = getGraphicsState().getStrokingColor().toRGB();
        } catch (Exception ex) {
        }
        String key = null;
        for (int i = 0; i < columns.size(); i++) {
            if (t.getX() <= columns.get(i)) {
                key = (i + "_");
                break;
            }
        }
        if (key == null) {
            key = (columns.size() + "_");
        }
        key += (t.getY() + "_" + t.getEndY() + "_" + t.getFontSize() + "_" + color);
        List<TextAndColor> arr = textItems.get(key);
        if (arr == null) {
            arr = new ArrayList<>();
        }
        arr.add(new TextAndColor(t, color));
        textItems.put(key, arr);
    }

    public static class XY implements Comparable<XY> {
        public static Float MIN_Y_DIFF = 0.005f;
        protected final Float x;
        protected final Float y;
        protected final Float h;
        protected final String fontName;
        protected final TextAndColor txt;

        public XY(TextAndColor txt, float x, float y, float h) {
            this.txt = txt;
            this.fontName = txt.getFont().getName();
            this.x = x;
            this.y = y;
            this.h = h;
            //System.out.println("~~~~~~~~~~~> " + txt.getUnicode() + " >> " + x + "," + y + "|" + h);
        }

        @Override
        public String toString() {
            return "'" + txt.getUnicode() + "' @ " + x + "," + y + "|" + h + " (" + fontName + ")";
        }

        @Override
        public int compareTo(XY o) {
            //System.out.print("### " + this.toString() + " ### " + o.toString() + " ###");
            int r = y.compareTo(o.y);
            if ((r == 0) || (Math.abs(y - o.y) < MIN_Y_DIFF)) {
                r = x.compareTo(o.x);
            }
            //if ((this.fontName != null) && (o.fontName != null))
            {
                //if (this.fontName.equals(o.fontName))
                {
                    float dx = this.x - o.x;
                    float dy = this.y - o.y;
                    float deltaH = (0.25f * (this.h + o.h));
                    if (((dx < 0.0) && (dy < 0.0)) || ((dx > 0.0) && (dy > 0.0))) {
                        r = this.x.compareTo(o.x);
                    } else if (Math.abs(this.y - o.y) > deltaH) {
                        r = this.y.compareTo(o.y);
                    } else if (Math.abs(this.x - o.x) > deltaH) {
                        r = this.x.compareTo(o.x);
                    } else {
                        float minYDiff = (0.1f * (this.h + o.h));
                        r = y.compareTo(o.y);
                        if ((r == 0) || (Math.abs(y - o.y) < minYDiff)) {
                            r = x.compareTo(o.x);
                        }
                    }
                }
            }
            //System.out.print(" ==> " + r + "\n");
            return r;
        }
    }

    public void addNewItem(String key, TextAndColor starter, String str, float maxY) {
        for (int i = 0; i < (columns.size() + 1); i++) {
            if (key.startsWith(i + "_")) {
                List<Pair<String, XY>> data = dataSet.get(i);
                if (data == null) {
                    data = new ArrayList<>();
                }
                data.add(new Pair<>(str, new XY(starter, starter.getX(), (maxY * i) + starter.getY(), starter.tp.getHeight())));
                dataSet.put(i, data);
                break;
            }
        }
    }

    public static String printString(List<TextAndColor> l) {
        List<TextAndColor> r = new ArrayList<>(l);
        r.sort((o1, o2) -> Float.compare(o1.getX(), o2.getX()));
        StringBuilder bld = new StringBuilder();
        for (TextAndColor t : r) {
            bld.append(t.getUnicode());
        }
        return bld.toString();
    }

    public StringFontColor getFirstTextLine(float maxY, boolean careAboutFont, boolean debug) {
        Set<String> rm = new HashSet<>();
        // Merge sub-supper scripts
        for (Map.Entry<String, List<TextAndColor>> entry : textItems.entrySet()) {
            String[] t1 = entry.getKey().split("_");
            int a_col = Integer.parseInt(t1[0]);
            float a_Y = Float.parseFloat(t1[1]);
            float a_Height = entry.getValue().get(0).getHeight();//Float.parseFloat(t1[3]);
            for (Map.Entry<String, List<TextAndColor>> subEntry : textItems.entrySet()) {
                if (!entry.getKey().equals(subEntry.getKey())) {
                    String[] t2 = subEntry.getKey().split("_");
                    int b_col = Integer.parseInt(t2[0]);
                    if (a_col == b_col) {
                        float b_Y = Float.parseFloat(t2[1]);
                        float b_Height = subEntry.getValue().get(0).getHeight();//Float.parseFloat(t2[3]);
                        float d;
                        if (b_Y < a_Y) {
                            d = a_Y - (b_Y + b_Height);
                        } else {
                            d = b_Y - (a_Y + a_Height);
                        }
                        if ((0.8 * a_Height) > b_Height) {
                            if (d < (0.6 * b_Height)) {
                                if (debug) {
                                    System.out.print("Transferring [d: " + d + ", h: " + b_Height + " @R: " + (d / b_Height) + " ]: (" + printString(subEntry.getValue()) + ")'" + subEntry.getKey() + "' to (" + printString(entry.getValue()) + ")'" + entry.getKey() + "'");
                                }
                                entry.getValue().addAll(subEntry.getValue());
                                rm.add(subEntry.getKey());
                                if (debug) {
                                    System.out.println("\n ==> (" + printString(entry.getValue()) + ")'");
                                }
                            }
                        }
                    }
                }
            }
        }
        for (String key : rm) {
            textItems.remove(key);
        }
        for (Map.Entry<String, List<TextAndColor>> entry : textItems.entrySet()) {
            List<TextAndColor> list = entry.getValue();
            list.sort((o1, o2) -> Float.compare(o1.getX(), o2.getX()));

            TextAndColor starter = list.get(0);
            String k1 = starter.getFontSize() + "_" + starter.getFont().getName();
            StringBuilder str = new StringBuilder(starter.getUnicode());
            for (int i = 1; i < list.size(); i++) {
                TextAndColor curr = list.get(i);
                String k2 = curr.getFontSize() + "_" + curr.getFont().getName();
                boolean subSupScript = false;
                float a_Y = starter.getY();
                float a_Height = starter.getHeight();
                float b_Y = curr.getY();
                float b_Height = curr.getHeight();
                float d;
                if (b_Y < a_Y) {
                    d = a_Y - (b_Y + b_Height);
                } else {
                    d = b_Y - (a_Y + a_Height);
                }
                if ((0.8 * a_Height) > b_Height) {
                    if (d < (0.6 * b_Height)) {
                        subSupScript = true;
                    }
                }
                if (k2.equals(k1) || subSupScript) {
                    str.append(curr.getUnicode());
                } else {
                    if (debug) {
                        System.out.println("Adding: '" + str.toString() + "' @ " + entry.getKey());
                    }
                    addNewItem(entry.getKey(), starter, str.toString(), maxY);
                    // Start afresh because we have a new font
                    starter = curr;
                    k1 = starter.getFontSize() + "_" + starter.getFont().getName();
                    str = new StringBuilder(starter.getUnicode());
                }
            }
            if (debug) {
                System.out.println("Adding: '" + str.toString() + "' @ " + entry.getKey());
            }
            addNewItem(entry.getKey(), starter, str.toString(), maxY);
        }
        List<Pair<String, XY>> data = dataSet.get(0);
        if (data != null) {
            StringBuilder bld = new StringBuilder();
            float minH = 9999999999999.9f;
            float maxElY = 0.0f;
            data.sort((o1, o2) -> o1.getValue().compareTo(o2.getValue()));
            Pair<String, XY> prev = null;
            for (Pair<String, XY> p : data) {
                boolean NotEmptySpace = (p.getKey().trim().length() > 0);
                boolean smallAddition = false;
                boolean onlyFontDifference = false;
                if (prev != null) {
                    String k1 = prev.getValue().txt.getFontSize() + "_" + prev.getValue().txt.color + "_" + prev.getValue().txt.getFont().getName();
                    String k2 = p.getValue().txt.getFontSize() + "_" + p.getValue().txt.color + "_" + p.getValue().txt.getFont().getName();

                    if (!careAboutFont) {
                        if (prev.getValue().y.floatValue() == p.getValue().y.floatValue()) {
                            onlyFontDifference = true;
                        }
                    }

                    float dist = ((p.getValue().y - prev.getValue().y) / prev.getValue().txt.getFontSize());
/*
                        smallAddition = ("º".equals(p.getKey().trim()) || "°".equals(p.getKey().trim()) || "ᵒ".equals(p.getKey().trim())) ||
                                (":".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (",".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (".".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (";".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("!".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("-".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("*".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("&".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold")));
*/
                    smallAddition = ("º".equals(p.getKey().trim()) || "°".equals(p.getKey().trim()) || "ᵒ".equals(p.getKey().trim())) ||
                            (":".equals(p.getKey().trim())) ||
                            (",".equals(p.getKey().trim())) ||
                            (".".equals(p.getKey().trim())) ||
                            (";".equals(p.getKey().trim())) ||
                            ("!".equals(p.getKey().trim())) ||
                            ("-".equals(p.getKey().trim())) ||
                            ("*".equals(p.getKey().trim())) ||
                            ("&".equals(p.getKey().trim()));

                    if (debug) {
                        System.out.println("Joining '" + p.getKey() + "' @ " + dist + " {" + p.getValue().y + " | " + p.getValue().h + "} >> '" + k1 + "' ~ '" + k2 + "'");
                    }
                    if ((!k1.equals(k2) || (dist > 2.0)) && NotEmptySpace && (!smallAddition) && (!onlyFontDifference)) {
                        return new StringFontColor(bld.toString().replaceAll("\\s+", " ").trim(),
                                prev.getValue().txt.getFontSize(),
                                prev.getValue().txt.getFont().getName(),
                                prev.getValue().txt.color,
                                prev.getValue().txt.getX(),
                                maxElY,//prev.getValue().txt.getY(),
                                minH);
                    }
                } else {
                    if (debug) {
                        String k2 = p.getValue().txt.getFontSize() + "_" + p.getValue().txt.color + "_" + p.getValue().txt.getFont().getName();
                        System.out.println("Starting '" + p.getKey() + "' @ " + p.getValue().y + " {" + p.getValue().y + " | " + p.getValue().h + "} '" + k2 + "'");
                    }
                }
                bld.append(p.getKey());
                minH = Math.min(minH, p.getValue().h);
                maxElY = Math.max(maxElY, p.getValue().y);
                if (NotEmptySpace && (!smallAddition)) {
                    prev = p;
                }
            }
            if (bld.length() > 0) {
                if (prev != null) {
                    return new StringFontColor(bld.toString().replaceAll("\\s+", " ").trim(),
                            prev.getValue().txt.getFontSize(),
                            prev.getValue().txt.getFont().getName(),
                            prev.getValue().txt.color,
                            prev.getValue().txt.getX(),
                            maxElY,//prev.getValue().txt.getY(),
                            minH);
                }
            }
        }
        return null;
    }

    public List<List<StringFontColor>> getTextColumns(float pageMinY, float pageMaxY, boolean careAboutFont, boolean debug) {
        Set<String> rm = new HashSet<>();
        // Merge sub-supper scripts
        for (Map.Entry<String, List<TextAndColor>> entry : textItems.entrySet()) {
            String[] t1 = entry.getKey().split("_");
            int a_col = Integer.parseInt(t1[0]);
            float a_Y = Float.parseFloat(t1[1]);
            float a_Height = entry.getValue().get(0).getHeight();//Float.parseFloat(t1[3]);
            if ((a_Y < pageMinY) || (a_Y > pageMaxY)) {
                rm.add(entry.getKey());
            }
        }
        for (String key : rm) {
            textItems.remove(key);
        }
        rm.clear();

        for (Map.Entry<String, List<TextAndColor>> entry : textItems.entrySet()) {
            String[] t1 = entry.getKey().split("_");
            int a_col = Integer.parseInt(t1[0]);
            float a_Y = Float.parseFloat(t1[1]);
            float a_Height = entry.getValue().get(0).getHeight();//Float.parseFloat(t1[3]);
            for (Map.Entry<String, List<TextAndColor>> subEntry : textItems.entrySet()) {
                if (!entry.getKey().equals(subEntry.getKey())) {
                    String[] t2 = subEntry.getKey().split("_");
                    int b_col = Integer.parseInt(t2[0]);
                    if (a_col == b_col) {
                        float b_Y = Float.parseFloat(t2[1]);
                        float b_Height = subEntry.getValue().get(0).getHeight();//Float.parseFloat(t2[3]);
                        float d;
                        if (b_Y < a_Y) {
                            d = a_Y - (b_Y + b_Height);
                        } else {
                            d = b_Y - (a_Y + a_Height);
                        }
                        if ((0.8 * a_Height) > b_Height) {
                            if (d < (0.6 * b_Height)) {
                                if (debug) {
                                    System.out.print("Transferring [d: " + d + ", h: " + b_Height + " @R: " + (d / b_Height) + " ]: (" + printString(subEntry.getValue()) + ")'" + subEntry.getKey() + "' to (" + printString(entry.getValue()) + ")'" + entry.getKey() + "'");
                                }
                                entry.getValue().addAll(subEntry.getValue());
                                rm.add(subEntry.getKey());
                                if (debug) {
                                    System.out.println("\n ==> (" + printString(entry.getValue()) + ")'");
                                }
                            }
                        }
                    }
                }
            }
        }
        for (String key : rm) {
            textItems.remove(key);
        }
        for (Map.Entry<String, List<TextAndColor>> entry : textItems.entrySet()) {
            List<TextAndColor> list = entry.getValue();
            list.sort((o1, o2) -> Float.compare(o1.getX(), o2.getX()));

            TextAndColor starter = list.get(0);
            String k1 = starter.getFontSize() + "_" + starter.getFont().getName();
            StringBuilder str = new StringBuilder(starter.getUnicode());
            for (int i = 1; i < list.size(); i++) {
                TextAndColor curr = list.get(i);
                String k2 = curr.getFontSize() + "_" + curr.getFont().getName();
                boolean subSupScript = false;
                float a_Y = starter.getY();
                float a_Height = starter.getHeight();
                float b_Y = curr.getY();
                float b_Height = curr.getHeight();
                float d;
                if (b_Y < a_Y) {
                    d = a_Y - (b_Y + b_Height);
                } else {
                    d = b_Y - (a_Y + a_Height);
                }
                if ((0.8 * a_Height) > b_Height) {
                    if (d < (0.6 * b_Height)) {
                        subSupScript = true;
                    }
                }
                if (k2.equals(k1) || subSupScript) {
                    str.append(curr.getUnicode());
                } else {
                    if (starter.getY() > pageMinY) {
                        if (debug) {
                            System.out.println("Adding: '" + str.toString() + "' @ " + entry.getKey());
                        }
                        addNewItem(entry.getKey(), starter, str.toString(), pageMaxY);
                    }
                    // Start afresh because we have a new font
                    starter = curr;
                    k1 = starter.getFontSize() + "_" + starter.getFont().getName();
                    str = new StringBuilder(starter.getUnicode());
                }
            }
            if (starter.getY() > pageMinY) {
                if (debug) {
                    System.out.println("Adding: '" + str.toString() + "' @ " + entry.getKey());
                }
                addNewItem(entry.getKey(), starter, str.toString(), pageMaxY);
            }
        }
        List<List<StringFontColor>> ret = new ArrayList<>();
        for (int col = 0; col < dataSet.size(); col++) {
            List<StringFontColor> textLines = new ArrayList<>();
            List<Pair<String, XY>> data = dataSet.get(col);
            if (data != null) {
                StringBuilder bld = new StringBuilder();
                float minH = 9999999999999.9f;
                float minY = 9999999999999.9f;
                data.sort((o1, o2) -> o1.getValue().compareTo(o2.getValue()));
                Pair<String, XY> prev = null;
                for (Pair<String, XY> p : data) {
                    if (debug)
                    {
                        XY v = p.getValue();
                        System.out.println("~~> Item: " + v.toString());
                    }
                    boolean NotEmptySpace = (p.getKey().trim().length() > 0);
                    boolean smallAddition = false;
                    boolean onlyFontDifference = false;
                    if (prev != null) {
                        String k1 = prev.getValue().txt.getFontSize() + "_" + prev.getValue().txt.color + "_" + prev.getValue().txt.getFont().getName();
                        String k2 = p.getValue().txt.getFontSize() + "_" + p.getValue().txt.color + "_" + p.getValue().txt.getFont().getName();

                        if (!careAboutFont) {
                            if (prev.getValue().y.floatValue() == p.getValue().y.floatValue()) {
                                onlyFontDifference = true;
                            }
                        }

                        float dist = ((p.getValue().y - prev.getValue().y) / prev.getValue().txt.getFontSize());
/*
                        smallAddition = ("º".equals(p.getKey().trim()) || "°".equals(p.getKey().trim()) || "ᵒ".equals(p.getKey().trim())) ||
                                (":".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (",".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (".".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                (";".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("!".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("-".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("*".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold"))) ||
                                ("&".equals(p.getKey().trim()) && (prev.getValue().txt.getFont().getName().toLowerCase().contains("bold")));
*/
                        smallAddition = ("º".equals(p.getKey().trim()) || "°".equals(p.getKey().trim()) || "ᵒ".equals(p.getKey().trim())) ||
                                (":".equals(p.getKey().trim())) ||
                                (",".equals(p.getKey().trim())) ||
                                (".".equals(p.getKey().trim())) ||
                                (";".equals(p.getKey().trim())) ||
                                ("!".equals(p.getKey().trim())) ||
                                ("-".equals(p.getKey().trim())) ||
                                ("*".equals(p.getKey().trim())) ||
                                ("&".equals(p.getKey().trim()));

                        if (debug) {
                            String myTxt = bld.toString().replaceAll("\\s+", " ").trim();
                            System.out.println("->Joining '" + p.getKey() + "' @ dist: " + dist + " {" + p.getValue().y + " | " + p.getValue().h + "} >> '" + k1 + "' ~ '" + k2 + "'");
//                            System.out.println("\tNotEmptySpace: " + NotEmptySpace +
//                                    ", !smallAddition: " + (!smallAddition) +
//                                    ", !onlyFontDifference: " + (!onlyFontDifference) +
//                                    ", (minY > pageMinY): " + (minY > pageMinY) +
//                                    ", (!k1.equals(k2) || (dist > 2.0)): " + ((!k1.equals(k2) || (dist > 2.0))));
//                            System.out.println("MyTxt: " + myTxt);
                        }
                        if ((!k1.equals(k2) || (dist > 2.0)) && NotEmptySpace && (!smallAddition) && (!onlyFontDifference)) {
                            if (minY > pageMinY) {//Skipping pageMinY
                                String text = bld.toString().replaceAll("\\s+", " ").trim();
                                if (debug) {
                                    System.out.println("==>Created: '" + text + "'");
                                }
                                textLines.add(new StringFontColor(text,
                                        prev.getValue().txt.getFontSize(),
                                        prev.getValue().txt.getFont().getName(),
                                        prev.getValue().txt.color,
                                        prev.getValue().txt.getX(),
                                        minY,//prev.getValue().txt.getY(),
                                        minH));
                            }
                            bld = new StringBuilder();
                            minH = 9999999999999.9f;
                            minY = 9999999999999.9f;
                        }
                    } else {
                        if (debug) {
                            String k2 = p.getValue().txt.getFontSize() + "_" + p.getValue().txt.color + "_" + p.getValue().txt.getFont().getName();
                            System.out.println("Starting '" + p.getKey() + "' @ " + p.getValue().y + " {" + p.getValue().y + " | " + p.getValue().h + "} '" + k2 + "'");
                        }
                    }
                    bld.append(p.getKey());
                    minH = Math.min(minH, p.getValue().h);
                    minY = Math.min(minY, p.getValue().y);
                    if (NotEmptySpace && (!smallAddition)) {
                        prev = p;
                    }
                }
                if (bld.length() > 0) {
                    if (prev != null) {
                        if (minY > pageMinY) {
                            textLines.add(new StringFontColor(bld.toString().replaceAll("\\s+", " ").trim(),
                                    prev.getValue().txt.getFontSize(),
                                    prev.getValue().txt.getFont().getName(),
                                    prev.getValue().txt.color,
                                    prev.getValue().txt.getX(),
                                    minY,//prev.getValue().txt.getY(),
                                    minH));
                        }
                    }
                }
            }
            ret.add(textLines);
        }
        return ret;
    }

    public static void testTextExtract(String file, int page, int expectedColumns, Set<String> fontFilter, boolean showDetails, boolean debug) {
        try {
            PDDocument doc = PDDocument.load(new File(file));
            PDPage pg = doc.getPage(page);
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(pg);
            List<Float> cols = det.getColumnSeparators(expectedColumns, debug);
            PDFTextReader read = new PDFTextReader(cols);
            read.processPage(pg);
            if (debug) {
                for (Float col : cols) {
                    System.out.println(">>>>>>>>> COL AT: " + col);
                }
            }
            List<List<StringFontColor>> text = read.getTextColumns(0.0f, pg.getCropBox().getHeight(), false, debug);
            for (int i = 0; i < text.size(); i++) {
                System.out.println("\n\n======================= COLUMN #" + (i + 1) + " =============\n");
                float y = 0;
                for (StringFontColor str : text.get(i)) {
                    if (fontFilter.isEmpty() || fontFilter.contains(str.fontName)) {
                        if ((str.y - y) > (str.fontSize / 2)) {
                            System.out.println();
                        }
                        if (showDetails) {
                            System.out.print("\t[" + str.text + "] (" + str.x + ", " + str.y + " | " + str.fontSize + " : " + str.color + " : " + str.fontName + ")");
                        } else {
                            System.out.print("\t[" + str.text + "]");
                        }
                        y = str.y;
                    }
                }
            }
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void testExtractFilteredText(String file, int page, int expectedColumns, Set<String> fontFilter, boolean showDetails, boolean debug) {
        try {
            PDDocument doc = PDDocument.load(new File(file));
            PDPage pg = doc.getPage(page);
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(pg);
            List<Float> cols = det.getColumnSeparators(expectedColumns, debug);
            PDFTextReader read = new PDFTextReader(cols);
            read.processPage(pg);
            if (debug) {
                for (Float col : cols) {
                    System.out.println(">>>>>>>>> COL AT: " + col);
                }
            }
            List<List<StringFontColor>> text = read.getTextColumns(0.0f, pg.getCropBox().getHeight(), false, debug);
            for (int i = 0; i < text.size(); i++) {
                System.out.println("\n\n======================= COLUMN #" + (i + 1) + " =============\n");
                float y = 0;
                for (StringFontColor str : text.get(i)) {
                    if ((fontFilter.isEmpty() || fontFilter.contains(str.fontName)) && (str.text != null) && (str.text.length() > 0)) {
                        if ((str.y - y) > (str.fontSize / 2)) {
                            System.out.println();
                        }
                        if (showDetails) {
                            System.out.print("\t[" + str.text + "] (" + str.x + ", " + str.y + " | " + str.fontSize + " : " + str.color + " : " + str.fontName + ")");
                        } else {
                            System.out.print("\t[" + str.text + "]");
                        }
                        y = str.y;
                    }
                }
            }
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        Set<String> fonts = new HashSet<>();
        fonts.add("VOWLNL+TTE3516960t00");
        fonts.add("ULAZAH+TTE3518A50t00");

        testExtractFilteredText("Habumuremyi.pdf", 404, 1, fonts, true, false);
    }

}
