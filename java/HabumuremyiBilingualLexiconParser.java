
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class HabumuremyiBilingualLexiconParser {
    public static final String HABUMUREMYI_FILE = "Habumuremyi.pdf";
    static int expectedCols = 2;
    static boolean debug = false;
    static float marginY = 60.0f;

    public static List<Pair<String, List<String>>> parseEntries(int meaningColor,
                                                                int startPageNumber, int endPageNumber) throws IOException {
        List<Pair<String, List<String>>> ret = new ArrayList<>();
        PDDocument doc = PDDocument.load(new File(HABUMUREMYI_FILE));
        int STATE = 0;
        String entry = null;
        List<String> meaningList = new ArrayList<>();
        StringBuilder meaning = new StringBuilder();
        long startMs = System.currentTimeMillis();
        int idx = 0;
        int total = endPageNumber - startPageNumber + 1;
        for (int pageNumber = startPageNumber; pageNumber <= endPageNumber; pageNumber++) {
            idx++;
            PDPage page = doc.getPage(pageNumber - 1);
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(page);
            List<Float> cols = det.getColumnSeparators(expectedCols, debug);
            PDFTextReader read = new PDFTextReader(cols);
            read.processPage(page);
            List<List<PDFTextReader.StringFontColor>> text = read.getTextColumns(marginY, page.getCropBox().getHeight() - marginY, true, debug);
            for (int i = 0; i < text.size(); i++) {
                for (PDFTextReader.StringFontColor str : text.get(i)) {
                    if (str.fontName != null) {
                        if (Math.abs(str.fontSize - 12.96f) < 0.01) {
                            if (str.fontName.startsWith("VOWLNL") && (str.color == 0)) {
                                if (!str.text.endsWith("•")) {
                                    if (STATE == 0) {
                                        entry = str.text.trim();
                                    } else if (STATE == 1) {
                                        entry = str.text.trim();
                                    } else {
                                        if (STATE == 3) {
                                            meaningList.add(meaning.toString().trim());
                                        }
                                        ret.add(new Pair<>(entry, meaningList));
                                        meaningList = new ArrayList<>();
                                        meaning = new StringBuilder();
                                        entry = str.text.trim();
                                    }
                                    // New lexicon entry
                                    // System.out.print("\n\"" + str.text + "\"");// + "\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                                    STATE = 1;
                                } else {
                                    if (STATE == 3) {
                                        meaningList.add(meaning.toString().trim());
                                        meaning = new StringBuilder();
                                    }
                                    // ==> Alternative meaning
                                    // System.out.println();
                                    STATE = 2;
                                }
                            } else if (str.fontName.startsWith("YARFYB") && (str.color == meaningColor)) {
                                if ((STATE == 1) || (STATE == 2) || (STATE == 3)) {
                                    meaning.append(" ").append(str.text.trim());
                                }
                                // Potential translation
                                // System.out.print("\t'" + str.text + "'");//+"\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                                STATE = 3;
                            }
                        }
                    }
                }
            }
            if (idx % 10 == 0) {
                System.out.println("Processed " + idx + "/" + total + " " + ETA.printETA(startMs, idx, total));
            }
        }
        doc.close();
        return ret;
    }

    static final Pattern subscribed = Pattern.compile("^[a-zA-Z][a-zA-Z]+[12345]$");

    public static List<String> processEntry(String entry) {
        List<String> ret = new ArrayList<>();
        if (entry != null) {
            // 1. Remove subscript
            if (subscribed.matcher(entry).find()) {
                entry = entry.substring(0, entry.length() - 1);
            }
            // 4. Split entries
            for (String en : entry.split(",")) {
                String v = en.trim();
                if (v.length() > 1) {
                    ret.add(v);
                }
            }
        }
        return ret;
    }

    // @TODO: Filtering ideas
    // 1. Remove subscript from entries, e.g. cyiza1, cyiza2, ... able to detect them.
    // 2. For each meaning, only copy until point/dot after gaining content. e.g. 'good; satisfactory in quality, quantity, or degree. a good teacher. good health.'
    //        ==> 'good; satisfactory in quality, quantity, or degree.' Stop right there befor cutting it into pieces.
    // 3. Split alternative meanings with ';' then trim.
    // 4. Split alternative entries with ',' then trim. e.g. Page 665: "conceraffect, touch, bear upoinvolve, be relevato, relate to, pertain to, appertain to, apply to, interest, occupy, be one's business" :
    // 5. Detect and remove long stories explaining in non-dictionary/encyclopedic ways, e.g. Kings of Rwanda,... Biblical names, e.g. Eva

    public static List<String> processMeaning(List<String> meaningList) {
        List<String> ret = new ArrayList<>();
        for (String mean : meaningList) {
            mean = mean.trim();
            // 2. Remove dot
            if (mean.startsWith(".")) {
                mean = mean.substring(1).trim();
            }
            int idx = mean.indexOf('.');
            if (idx > 0) {
                mean = mean.substring(0, idx).trim();
            }
            // 3. Split by ';' and trim
            // 5. Detect long stories
            String[] tk = mean.split(";");
            boolean keep = true;
            if (tk.length > 0) {
                if (tk[0].length() > 0) {
                    keep = tk[0].trim().split(" ").length <= 4;
                }
            }
            if (keep) {
                for (String t : tk) {
                    if (t.length() > 0) {
                        if (t.trim().split(" ").length <= 4) {
                            String v = t.trim();
                            if (v.length() > 1) {
                                ret.add(v);
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }
}
