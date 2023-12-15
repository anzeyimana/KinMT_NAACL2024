
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class HabumuremyiBilingualLexiconParserTest {
    public static final String HABUMUREMYI_FILE = "Habumuremyi.pdf";

    @Test
    public void test_RW_EN() {
        // Kinyarwanda -> English: 36 - 558
        int expectedCols = 2;
        int pageNumber = 38;
        boolean debug = false;
        float marginY = 60.0f;
        try {
            PDDocument doc = PDDocument.load(new File(HABUMUREMYI_FILE));
            PDPage page = doc.getPage(pageNumber - 1);
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(page);
            List<Float> cols = det.getColumnSeparators(expectedCols, debug);
            PDFTextReader read = new PDFTextReader(cols);
            read.processPage(page);
            List<List<PDFTextReader.StringFontColor>> text = read.getTextColumns(marginY, page.getCropBox().getHeight() - marginY, true, debug);
            for (int i = 0; i < text.size(); i++) {
                if (debug) {
                    System.out.println("\n\n======================= COLUMN #" + (i + 1) + " =============\n");
                }
                for (PDFTextReader.StringFontColor str : text.get(i)) {
                    if ((str.color == 0) && (str.fontName != null)) {
                        if (Math.abs(str.fontSize - 12.96f) < 0.01) {
                            if (str.fontName.startsWith("VOWLNL")) {
                                if (!str.text.endsWith("•")) {
                                    // New lexicon entry
                                    System.out.print("\n\"" + str.text + "\"");// + "\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                                } else {
                                    // ==> Alternative meaning
                                    System.out.println();
                                }
                            } else if (str.fontName.startsWith("YARFYB")) {
                                // Potential translation
                                System.out.print("\t'" + str.text + "'");//+"\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                            } else {
                                //    System.out.println("\t*" + str.text + "*\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                            }
                        }
                    }
                }
            }
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        System.out.println("\n");
    }

    @Test
    public void test_EN_RW() {
        // Kinyarwanda -> English: 563 - 1058
        int expectedCols = 2;
        int pageNumber = 1058;
        boolean debug = false;
        float marginY = 60.0f;
        try {
            PDDocument doc = PDDocument.load(new File(HABUMUREMYI_FILE));
            PDPage page = doc.getPage(pageNumber - 1);
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(page);
            List<Float> cols = det.getColumnSeparators(expectedCols, debug);
            PDFTextReader read = new PDFTextReader(cols);
            read.processPage(page);
            List<List<PDFTextReader.StringFontColor>> text = read.getTextColumns(marginY, page.getCropBox().getHeight() - marginY, true, debug);
            for (int i = 0; i < text.size(); i++) {
                if (debug) {
                    System.out.println("\n\n======================= COLUMN #" + (i + 1) + " =============\n");
                }
                for (PDFTextReader.StringFontColor str : text.get(i)) {
                    //System.out.println("==> '" + str.text + "'\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                    if ((str.fontName != null)) {
                        if (Math.abs(str.fontSize - 12.96f) < 0.01) {
                            if (str.fontName.startsWith("VOWLNL") && (str.color == 0)) {
                                if (!str.text.endsWith("•")) {
                                    // New lexicon entry
                                    //System.out.print("\n\"" + str.text + "\"");// + "\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);

                                    // Alternative inputs:
                                    String[] toks = str.text.split(",");
                                    for (String tk : toks) {
                                        System.out.print("\n\"" + tk.trim() + "\"");
                                    }
                                } else {
                                    // ==> Alternative meaning
                                    System.out.println();
                                }
                            } else if (str.fontName.startsWith("YARFYB") && (str.color == 128)) {
                                // Potential translation
                                System.out.print("\t'" + str.text + "'");//+"\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                            } else {
                                //System.out.println("\t*" + str.text + "*\t\t\t\t~~~~> " + str.x + "," + str.y + ": " + str.text + "/" + str.color + "/" + str.fontName + "/" + str.fontSize);
                            }
                        }
                    }
                }
            }
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        System.out.println("\n");
    }

    @Test
    public void testGenericParse_RW_EN() throws Exception {
        List<Pair<String, List<String>>> entries = HabumuremyiBilingualLexiconParser.parseEntries(0, 36, 558);
//        List<Pair<String, List<String>>> entries = HabumuremyiBilingualLexiconParser.parseEntries(0, 241, 243);
        for (Pair<String, List<String>> en : entries) {
            System.out.println("\"" + en.getKey() + "\" :");
            for (String mean : en.getValue()) {
                System.out.println("\t'" + mean + "'");
            }
            System.out.println("==>E: " + Arrays.toString(HabumuremyiBilingualLexiconParser.processEntry(en.getKey()).toArray()));
            System.out.println("~~>M: " + Arrays.toString(HabumuremyiBilingualLexiconParser.processMeaning(en.getValue()).toArray()));
            System.out.println();
        }
    }

    @Test
    public void testGenericParse_EN_RW() throws Exception {
        List<Pair<String, List<String>>> entries = HabumuremyiBilingualLexiconParser.parseEntries(128, 563, 1058);
        for (Pair<String, List<String>> en : entries) {
            System.out.println("\"" + en.getKey() + "\" :");
            for (String meaning : en.getValue()) {
                System.out.println("\t'" + meaning + "'");
            }
            System.out.println("==>E: " + Arrays.toString(HabumuremyiBilingualLexiconParser.processEntry(en.getKey()).toArray()));
            System.out.println("~~>M: " + Arrays.toString(HabumuremyiBilingualLexiconParser.processMeaning(en.getValue()).toArray()));
            System.out.println();
        }
    }


    // @TODO: Filtering ideas
    // 1. Remove subscript from entries, e.g. cyiza1, cyiza2, ... able to detect them.
    // 2. For each meaning, only copy until point/dot after gaining content. e.g. 'good; satisfactory in quality, quantity, or degree. a good teacher. good health.'
    //        ==> 'good; satisfactory in quality, quantity, or degree.' Stop right there befor cutting it into pieces.
    // 3. Split alternative meanings with ';' then trim.
    // 4. Split alternative entries with ',' then trim. e.g. Page 665: "conceraffect, touch, bear upoinvolve, be relevato, relate to, pertain to, appertain to, apply to, interest, occupy, be one's business" :
    // 5. Detect and remove long stories explaining in non-dictionary/encyclopedic ways, e.g. Kings of Rwanda,... Biblical names, e.g. Eva

    public static void main(String[] args) throws Exception {
        PrintWriter combined_rw = new PrintWriter(new FileWriter("habumuremyi_combined_rw.txt"), true);
        PrintWriter combined_en = new PrintWriter(new FileWriter("habumuremyi_combined_en.txt"), true);

        PrintWriter rw_en = new PrintWriter(new FileWriter("habumuremyi_rw_en.tsv"), true);
        List<Pair<String, List<String>>> kinyaEntries = HabumuremyiBilingualLexiconParser.parseEntries(0, 36, 558);
        for (Pair<String, List<String>> en : kinyaEntries) {
            List<String> entries = HabumuremyiBilingualLexiconParser.processEntry(en.getKey());
            List<String> meanings = HabumuremyiBilingualLexiconParser.processMeaning(en.getValue());
            for (String entry : entries) {
                for (String meaning : meanings) {
                    String val = meaning.replace("\t", " ").trim();
                    val = val.replace("(BrE)", "").trim();
                    val = val.replace("(AmE)", "").trim();

                    entry = PlainTextProcessor.normalizeKinyarwanda(entry);
                    val = PlainTextProcessor.normalizeEnglish(val);

                    if ((entry.length() > 1) && (val.length() > 1)) {
                        rw_en.println(entry + "\t" + val);
                        combined_rw.println(entry);
                        combined_en.println(val);
                    }
                }
            }
        }
        rw_en.close();

        PrintWriter en_rw = new PrintWriter(new FileWriter("habumuremyi_en_rw.tsv"), true);
        List<Pair<String, List<String>>> englishEntries = HabumuremyiBilingualLexiconParser.parseEntries(128, 563, 1058);
        for (Pair<String, List<String>> en : englishEntries) {
            List<String> entries = HabumuremyiBilingualLexiconParser.processEntry(en.getKey());
            List<String> meanings = HabumuremyiBilingualLexiconParser.processMeaning(en.getValue());
            for (String entry : entries) {
                entry = entry.replace("(BrE)", "").trim();
                entry = entry.replace("(AmE)", "").trim();
                for (String meaning : meanings) {
                    String val = meaning.replace("\t", " ").trim();

                    val = PlainTextProcessor.normalizeKinyarwanda(val);
                    entry = PlainTextProcessor.normalizeEnglish(entry);

                    if ((entry.length() > 1) && (val.length() > 1)) {
                        en_rw.println(entry + "\t" + val);
                        combined_rw.println(val);
                        combined_en.println(entry);
                    }
                }
            }
        }
        en_rw.close();

        combined_rw.close();
        combined_en.close();
    }
}
