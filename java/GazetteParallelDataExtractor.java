
import com.github.pemistahl.lingua.api.Language;
import com.github.pemistahl.lingua.api.LanguageDetector;
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class GazetteParallelDataExtractor {

    public static final LanguageDetector languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.FRENCH).build();

    public static class LabeledStringFontColor extends PDFTextReader.StringFontColor {

        LabeledStringFontColor prevElement = null;

        public final String prevNullReason;

        public ArticleParagraph paragraph = null;

        public String language = null;

        boolean isOffifialGazetteMark = false;

        boolean isArticleTitle = false;

        final int page;

        int articleNumber = 0;
        int column = 0;

        boolean isSectionTitle = false;

        boolean isBlank = false;
        boolean isSignature = false;
        boolean isChapterTitle = false;


        boolean isDoneMarker = false;

        boolean isOrderMarker = false;

        boolean isPageMark = false;

        boolean isArticleParagraph = false;
        boolean isArticleParagraphComplete = false;

        boolean isUnknownType = false;

        static final Pattern OfficialGazette = Pattern.compile("(official\\s+gazette)|(o\\.g\\.)");
        static final Pattern No = Pattern.compile("(nº)|(no)|(n°)|(nᵒ)|(n\\u00BA)");
        static final Pattern Date = Pattern.compile("(\\d{1,2}/\\d{1,2}/\\d{4})");
        static final Pattern MonthEn = Pattern.compile("(jan(?:uary)?)|(feb(?:ruary)?)|(mar(?:ch)?)|(apr(?:il)?)|(may)|(jun(?:e)?)|(jul(?:y)?)|(aug(?:ust)?)|(sep(?:tember)?)|(oct(?:ober)?)|(nov(?:ember)?)|(dec(?:ember)?)");
        static final Pattern Year = Pattern.compile("(\\d{4})");

        static final Pattern DoneOn = Pattern.compile("(ku\\s?wa\\s?\\.?\\s?)|(on\\s?\\.?\\s?)|(le\\s?\\.?\\s?)");
        static final Pattern MonthAny = Pattern.compile("(jan(?:uary)?)|(feb(?:ruary)?)|(mar(?:ch)?)|(apr(?:il)?)|(may)|(jun(?:e)?)|(jul(?:y)?)|(aug(?:ust)?)|(sep(?:tember)?)|(oct(?:ober)?)|(nov(?:ember)?)|(dec(?:ember)?)|(mutarama)|(gashyantare)|(werurwe)|(mata)|(gicurasi)|(kamena)|(nyakanga)|(kanama)|(nze[lr]i)|(ukwakira)|(ugushyingo)|(ukuboza)|(janvier)|(f((é)|e)vrier)|(mars)|(avril)|(mai)|(juin)|(juillet)|(ao((û)|u)t)|(septembre)|(octobre)|(novembre)|(d((é)|e)cembre)");

        static final Pattern Number = Pattern.compile("^\\d+$");

        static final Pattern ArticleTitleKinya = Pattern.compile("^ing[ie]n?go?\\s?\\.?\\s?(ya)?\\s?((mbere)|(\\d+))\\s?\\.?\\s?\\:?");
        static final Pattern ArticleTitleEnglish = Pattern.compile("^article?s?\\s?\\.?\\s?(ya)?\\s?((one)|(\\d+))\\s?\\.?\\s?\\:?");
        static final Pattern ArticleTitleFrench = Pattern.compile("^article?s?\\s?\\.?\\s?(ya)?\\s?((premier)|(\\d+))\\s?\\.?\\s?\\:?");

        static final Pattern ArticleNoExtractKinya = Pattern.compile("(?<=^ing[ie]n?go?\\s?\\.?\\s?(ya)?\\s?)((mbere)|(\\d+))(?=\\s?\\.?\\s?\\:?)");
        static final Pattern ArticleNoExtractEnglish = Pattern.compile("(?<=^article?s?\\s?\\.?\\s?(ya)?\\s?)((one)|(\\d+))(?=\\s?\\.?\\s?\\:?)");
        static final Pattern ArticleNoExtractFrench = Pattern.compile("(?<=^article?s?\\s?\\.?\\s?(ya)?\\s?)((premier)|(\\d+))(?=\\s?\\.?\\s?\\:?)");

        static final Pattern SectionTitleKinya = Pattern.compile("^i((((cy)|k)iciro?)|(gice?))\\s?\\.?\\s?(cya)?\\s?((mbere)|([mdclxvi0-9]+)|(\\d+))\\s?\\.?\\s?\\:?");
        static final Pattern SectionTitleEnglish = Pattern.compile("^section?\\s?\\.?\\s?((one)|(two)|(three)|(four)|(five)|([mdclxvi0-9]+)|(\\d+))\\s?\\.?\\s?\\:?");
        static final Pattern SectionTitleFrench = Pattern.compile("^section?\\s?\\.?\\s?((premi((è)|e)re)|(deuxi((è)|e)me)|(troisi((è)|e)me)|(quatri((è)|e)me)|(cinqu?i((è)|e)me)|([mdclxvi0-9]+)|(\\d+))\\s?\\.?\\s?\\:?");

        static final Pattern Bold = Pattern.compile("bold");

        static final Pattern ChapterTitleKinya = Pattern.compile("^[UI]M[UI]?TW?E?\\s?([WY]A)?\\s?\\.?\\s?[MDCLXVI0-9]+\\s?\\.?\\s?\\:?");
        static final Pattern ChapterTitleEnglish = Pattern.compile("^CHA?PI?TERS?\\s?\\.?\\s?[MDCLXVI0-9]+\\s?\\.?\\s?\\:?");
        static final Pattern ChapterTitleFrench = Pattern.compile("^CHA?PI?TRES?\\s?\\.?\\s?[MDCLXVI0-9]+\\s?\\.?\\s?\\:?");

        static final Pattern Signature = Pattern.compile("^\\s?\\(\\s?s((e)|(é))\\s?\\)\\s?");

        static final Pattern OrderKinya = Pattern.compile("^(B?ATEGETSE)|([BY]?EMEJE)");
        static final Pattern OrderEnglish = Pattern.compile("^(HEREBY\\s?)?((ORDERS?)|(ADOPTS?))");
        static final Pattern OrderFrench = Pattern.compile("^(ARRETE(NT)?)|(ADOPTE(NT)?)");

        static final Pattern DoneKinya = Pattern.compile("ku\\s?wa\\s?\\.?\\s?(\\d{1,2}\\s?[/\\.\\-]\\s?\\d{1,2}\\s?[/\\.\\-]\\s?\\d{4})\\s?\\.?\\s?$");
        static final Pattern DoneEnglish = Pattern.compile("on\\s?\\.?\\s?(\\d{1,2}\\s?[/\\.\\-]\\s?\\d{1,2}\\s?[/\\.\\-]\\s?\\d{4})\\s?\\.?\\s?$");
        static final Pattern DoneFrench = Pattern.compile("le\\s?\\.?\\s?(\\d{1,2}\\s?[/\\.\\-]\\s?\\d{1,2}\\s?[/\\.\\-]\\s?\\d{4})\\s?\\.?\\s?$");

        public LabeledStringFontColor(int page, int col, List<PDFTextReader.StringFontColor> list, int idx, LabeledStringFontColor prev, String prevNullReason) {
            super(list.get(idx).text, list.get(idx).fontSize, list.get(idx).fontName, list.get(idx).color, list.get(idx).x, list.get(idx).y, list.get(idx).h);
            this.prevNullReason = prevNullReason;
            this.prevElement = prev;
            int tokensNum = text.trim().split("\\s+").length;
            this.column = col;
            this.page = page;
            if (prev != null) {
                language = prev.language;
            }
            isOffifialGazetteMark = (idx == 0) &&
                    (OfficialGazette.matcher(text.toLowerCase()).find()) &&
                    (No.matcher(text.toLowerCase()).find()) &&
                    (Date.matcher(text.toLowerCase()).find() || (MonthEn.matcher(text.toLowerCase()).find() && Year.matcher(text.toLowerCase()).find()));

            isArticleTitle = ((ArticleTitleKinya.matcher(text.toLowerCase().trim()).find() /*&& (par == 0)*/) ||
                    (ArticleTitleEnglish.matcher(text.toLowerCase().trim()).find() /*&& (par == 1)*/) ||
                    (ArticleTitleFrench.matcher(text.toLowerCase().trim()).find() /*&& (par == 2)*/)) &&
                    Bold.matcher(fontName.toLowerCase().trim()).find();

            if (isArticleTitle) {
                String num = "0";
                String l = null;
                Matcher matcher = ArticleNoExtractKinya.matcher(text.trim().toLowerCase());
                if (matcher.find()) {
                    num = matcher.group();
                    l = "rw";
                } else {
                    matcher = ArticleNoExtractEnglish.matcher(text.trim().toLowerCase());
                    if (matcher.find()) {
                        num = matcher.group();
                        l = "fe";
                    } else {
                        matcher = ArticleNoExtractFrench.matcher(text.trim().toLowerCase());
                        if (matcher.find()) {
                            num = matcher.group();
                            l = "fe";
                        }
                    }
                }
                if ("mbere".equals(num)) {
                    language = "rw";
                } else if ("one".equals(num)) {
                    language = "en";
                } else if ("premier".equals(num)) {
                    language = "fr";
                } else {
                    language = l;
                    if ("fe".equals(l) && (prev != null) && (!"1".equals(num))) {
                        if ("fr".equals(prev.language)) {
                            language = "fr";
                        } else if ("en".equals(prev.language)) {
                            language = "en";
                        }
                    }
                }
                if ("mbere".equals(num) || "one".equals(num) || "premier".equals(num)) {
                    articleNumber = 1;
                } else {
                    try {
                        articleNumber = Integer.parseInt(num);
                    } catch (Exception ex) {
                    }
                }
            }

            if (isArticleTitle && (prev != null)) {
                if (prev.isArticleTitle) {
                    if ((prev.articleNumber != (articleNumber - 1))) {
                        isArticleTitle = false;
                    }
                }
            }

            isSectionTitle = ((SectionTitleKinya.matcher(text.toLowerCase().trim()).find() /*&& (par == 0)*/) ||
                    (SectionTitleEnglish.matcher(text.toLowerCase().trim()).find() /*&& (par == 1)*/) ||
                    (SectionTitleFrench.matcher(text.toLowerCase().trim()).find() /*&& (par == 2)*/)) &&
                    Bold.matcher(fontName.toLowerCase().trim()).find();

            isChapterTitle = ((ChapterTitleKinya.matcher(text.trim()).find() /*&& (par == 0)*/) ||
                    (ChapterTitleEnglish.matcher(text.trim()).find() /*&& (par == 1)*/) ||
                    (ChapterTitleFrench.matcher(text.trim()).find() /*&& (par == 2)*/)) &&
                    Bold.matcher(fontName.toLowerCase().trim()).find();

            isOrderMarker = ((OrderKinya.matcher(text.trim()).find() /*&& (par == 0)*/) ||
                    (OrderEnglish.matcher(text.trim()).find() /*&& (par == 1)*/) ||
                    (OrderFrench.matcher(text.trim()).find() /*&& (par == 2)*/)) &&
                    Bold.matcher(fontName.toLowerCase().trim()).find() && (tokensNum < 5);

            isDoneMarker = ((DoneKinya.matcher(text.toLowerCase().trim()).find() /*&& (par == 0)*/) ||
                    (DoneEnglish.matcher(text.toLowerCase().trim()).find() /*&& (par == 1)*/) ||
                    (DoneFrench.matcher(text.toLowerCase().trim()).find() /*&& (par == 2)*/) ||
                    (DoneOn.matcher(text.toLowerCase().trim()).find() && MonthAny.matcher(text.toLowerCase().trim()).find() && Year.matcher(text.toLowerCase().trim()).find())) &&
                    (tokensNum < 10);// && (idx < (list.size() - 1));

            isBlank = (text.trim().replaceAll("\\s+", "").length() == 0);

            isSignature = Signature.matcher(text.trim().toLowerCase()).find();// && (tokensNum == 1);

            isPageMark = Number.matcher(text.trim()).find() && (idx == (list.size() - 1)) && (tokensNum == 1);

            isArticleParagraph = (!(isOffifialGazetteMark || isArticleTitle || isSectionTitle || isChapterTitle || isOrderMarker || isDoneMarker || isSignature || isPageMark));
            if (prev != null) {
                isArticleParagraph = isArticleParagraph && (prev.isArticleTitle || prev.isArticleParagraph);
                if (isArticleParagraph) {
                    articleNumber = prev.articleNumber;
                    prevElement = prev;
                    language = prev.language;
                    if (prev.isArticleTitle) {
                        if (text.trim().startsWith(":") || (y <= prev.y)) {
                            isArticleParagraph = false;
                            isArticleTitle = true;
                        }
//                        else {
//                            System.out.println("\nprev @ y: " + prev.y + " | h: " + prev.h + " ::: " + prev.labeledString());
//                            System.out.println("curr @ y: " + y + " | h: " + h + " | d: " + Math.abs(y - prev.y) + " ::: " + labeledString() + "\n");
//                        }
                    }
                    if (prev.isArticleTitle && Bold.matcher(fontName.toLowerCase().trim()).find()) {
                        isArticleParagraph = false;
                    }
                }
            } else {
                isArticleParagraph = false;
            }
            if (isArticleParagraph) {
                isArticleParagraphComplete = text.trim().endsWith(".");
            }

            isUnknownType = !(isOffifialGazetteMark || isArticleTitle || isSectionTitle || isChapterTitle || isOrderMarker || isDoneMarker || isBlank || isSignature || isPageMark || isArticleParagraph);

        }

        public String label() {
            StringBuilder bld = new StringBuilder();
            if (isArticleTitle) {
                bld.append("[ARTICLE-TITLE #" + articleNumber + "] ");
            }
            if (isSectionTitle) {
                bld.append("[SECTION-TITLE] ");
            }
            if (isChapterTitle) {
                bld.append("[CHAPTER-TITLE] ");
            }
            if (isBlank) {
                bld.append("[BLANK] ");
            }
            if (isArticleParagraph) {
                bld.append("[PARAGRAPH @" + articleNumber + "] ");
                if (isArticleParagraphComplete) {
                    bld.append("[COMPLETE] ");
                } else {
                    bld.append("[INCOMPLETE] ");
                }
            }
            if (isSignature) {
                bld.append("[SIGNATURE] ");
            }
            if (isDoneMarker) {
                bld.append("[DONE] ");
            }
            if (isOrderMarker) {
                bld.append("[ORDER] ");
            }
            if (isOffifialGazetteMark) {
                bld.append("[GAZETTE] ");
            }
            if (isPageMark) {
                bld.append("[PAGE] ");
            }
            if (isUnknownType) {
                bld.append("[OTHER] ");
            }
            return bld.toString().trim();
        }

        public String labeledString() {
            return (label().trim() + " " + text.trim()).trim();
        }

        public String details() {
            StringBuilder bld = new StringBuilder();
            bld.append(labeledString());
            bld.append("\n~~~~~~~~~~~~~~~~>>> @(x: " + x + ", y: " + y + "): fontSize{" + fontSize + "} fontName{" + fontName + "} color{" + color + "}");
            return bld.toString();
        }
    }

    public static String paragraphKey(int sectionNum, int articleNum, int paragraphNum, int langCol, String language) {
        return String.format("%04d_%04d_%04d_%04d:%s", sectionNum, articleNum, paragraphNum, langCol, language);
    }

    public static String commonParagraphKey(int sectionNum, int articleNum, int paragraphNum) {
        return String.format("%04d_%04d_%04d", sectionNum, articleNum, paragraphNum);
    }

    public static String paragraphKey(ArticleParagraph p) {
        return paragraphKey(p.sectionNum, p.articleNum, p.paragraphNum, p.column, p.language);
    }

    public static String commonParagraphKey(ArticleParagraph p) {
        return commonParagraphKey(p.sectionNum, p.articleNum, p.paragraphNum);
    }

    public static class ArticleParagraph {
        public String text;
        public final int sectionNum;
        public final int page;
        public final int column;
        public final int articleNum;
        public final int paragraphNum;
        public String language;

        public ArticleParagraph(int page, String text, int sectionNum, int langCol, int articleNum, int paragraphNum, String language) {
            this.page = page;
            this.text = text;
            this.sectionNum = sectionNum;
            this.column = langCol;
            this.articleNum = articleNum;
            this.paragraphNum = paragraphNum;
            this.language = language;
            if ("fe".equals(language)) {
                try {
                    String lang = languageDetector.detectLanguageOf(text.toLowerCase().trim()).getIsoCode639_1().toString();
                    if ("fr".equals(lang) || "en".equals(lang)) {
                        this.language = lang;
                    }
                } catch (Exception ex) {
                }
            }
        }

        public String key() {
            return paragraphKey(this);
        }

        public String commonKey() {
            return commonParagraphKey(this);
        }

        public String append(String t) {
            if (t != null) {
                text = text.trim() + " " + t.trim();
            }
            return text;
        }
    }

    public static Map<String, ArticleParagraph> extractParagraphs(String inPdfFile) {
        Map<String, ArticleParagraph> ret = new TreeMap<>();
        try {
            PDDocument doc = PDDocument.load(new File(inPdfFile));

            List<List<List<LabeledStringFontColor>>> documentElements = new ArrayList<>();
            int start_page = 0;
            int end_page = doc.getNumberOfPages();
            boolean debug = false;
            for (int pg = start_page; pg < end_page; pg++) {
                PDPage page = doc.getPage(pg);
                PDFColumnDetector det = new PDFColumnDetector();
                det.processPage(page);

                List<Float> cols = det.getColumnSeparators(2, false);

                float pageMinY = 0.0f;
                PDFTextReader firstReader = new PDFTextReader(new ArrayList<Float>());
                firstReader.processPage(page);
                PDFTextReader.StringFontColor firstLine = firstReader.getFirstTextLine(page.getCropBox().getHeight(), false, false);
                if (firstLine != null) {
                    String text = firstLine.text.toLowerCase().trim();
                    if (debug) {
                        //System.out.println("\n\nfirstLine @Page # " + (pg + 1) + " >>>>>>>>>> " + text + "\n\n");
                    }
                    if ((LabeledStringFontColor.OfficialGazette.matcher(text.toLowerCase()).find()) &&
                            (LabeledStringFontColor.No.matcher(text.toLowerCase()).find()) &&
                            (LabeledStringFontColor.Date.matcher(text.toLowerCase()).find() || (LabeledStringFontColor.MonthEn.matcher(text.toLowerCase()).find() && LabeledStringFontColor.Year.matcher(text.toLowerCase()).find()))) {
                        pageMinY = firstLine.y + firstLine.h;
                        if (debug) {
                            //System.out.println("\n\npageMinY @Page # " + (pg + 1) + " >>>>>>>>>> " + pageMinY + "\n\n");
                        }
                    }
                }

                PDFTextReader read = new PDFTextReader(cols);
                read.processPage(page);

                List<List<PDFTextReader.StringFontColor>> paragraphs = read.getTextColumns(pageMinY, page.getCropBox().getHeight(), false, false);
                List<List<LabeledStringFontColor>> pageParagraphs = new ArrayList<>();
                if ((paragraphs.size() == 2) || paragraphs.size() == 3) {
                    //System.out.println("\n\n-------------------------------------------------------------------------------------------------------------\n\n");
                    for (int col = 0; col < paragraphs.size(); col++) {
                        List<LabeledStringFontColor> paragraphElements = new ArrayList<>();
                        List<PDFTextReader.StringFontColor> list = paragraphs.get(col);
                        LabeledStringFontColor prev = null;
                        String prevNullReason = "0";
                        if ((pg > start_page) && (documentElements.size() > 0)) {
                            List<List<LabeledStringFontColor>> pp = documentElements.get(documentElements.size() - 1);
                            if ((pp.size() == paragraphs.size())) {
                                List<LabeledStringFontColor> pe = pp.get(col);
                                if (pe.size() > 0) {
                                    LabeledStringFontColor el = pe.get(pe.size() - 1);
                                    if (el.isPageMark) {
                                        if (pe.size() > 1) {
                                            prev = pe.get(pe.size() - 2);
                                        } else {
                                            prevNullReason = "4";
                                        }
                                    } else {
                                        prev = el;
                                    }
                                } else {
                                    prevNullReason = "3";
                                }
                            } else {
                                prevNullReason = "2";
                            }
                        } else {
                            prevNullReason = ("1==> pg=" + pg + " start_page=" + start_page + " documentElements.size()=" + documentElements.size());
                        }
                        for (int idx = 0; idx < list.size(); idx++) {
                            LabeledStringFontColor el = new LabeledStringFontColor(pg, col, list, idx, prev, prevNullReason);
                            paragraphElements.add(el);
                            if (!(el.isOffifialGazetteMark || el.isPageMark)) {
                                prev = el;
                                prevNullReason = "OK:" + el.labeledString();
                            } else {
                                prevNullReason = "prevIsGazetteOrPageMark:" + el.labeledString();
                            }
                            //System.out.println(s.text);
                            //System.out.println("===================> @(" + s.x + "," + s.y + "): color{" + s.color + "} fontSize{" + s.fontSize + "} fontName{" + s.fontName + "}");
                            //System.out.println(s.text + " @(" + s.x + "," + s.y + "): color{" + s.color + "} fontSize{" + s.fontSize + "} fontName{" + s.fontName + "}");
                        }
                        pageParagraphs.add(paragraphElements);
                        //System.out.println("\n\n-------------------------------------------------------------------------------------------------------------\n\n");
                    }
                }
                documentElements.add(pageParagraphs);
                //System.out.println("\n\n-------------------------------------------------------------------------------------------------------------\n\n");
            }
            int section = 0;
            int myPage = 0;
            for (List<List<LabeledStringFontColor>> pp : documentElements) {
                myPage++;
                if (debug) {
                    System.out.println("\n\n===================== PAGE " + (start_page + myPage) + " ===========================");
                }
                for (List<LabeledStringFontColor> pe : pp) {
                    if (debug) {
                        System.out.println("\n-------------------------------------------------------------------------------------------------------------\n");
                    }
                    for (LabeledStringFontColor el : pe) {
                        if (el.isArticleParagraph && (el.articleNumber > 0)) {
                            boolean attach = false;
                            if (el.prevElement != null) {
                                if (el.prevElement.isArticleParagraph && (el.prevElement.articleNumber > 0)) {
                                    if (((el.prevElement.articleNumber == el.articleNumber) &&
                                            (!el.prevElement.isArticleParagraphComplete) &&
                                            (el.prevElement.paragraph != null)) ||
                                            ((el.prevElement.articleNumber == el.articleNumber) && (el.y.floatValue() == el.prevElement.y.floatValue()))) {
                                        attach = true;
                                        el.paragraph = el.prevElement.paragraph;
                                        el.paragraph.append(el.text);
                                    }
                                }
                            }
                            if (!attach) {
                                int paragraph = 1;
                                if (el.prevElement != null) {
                                    if (el.prevElement.isArticleParagraph && (el.prevElement.paragraph != null)) {
                                        paragraph = el.prevElement.paragraph.paragraphNum + 1;
                                    }
                                }

                                if ((el.column == 0) && (paragraph == 1) && (el.articleNumber == 1)) {
                                    section++;
                                }
                                ArticleParagraph p = new ArticleParagraph(el.page, el.text, section, el.column, el.articleNumber, paragraph, el.language);
                                el.paragraph = p;
                                if (ret.containsKey(p.key())) {
                                    System.out.println("\n\nPAGE # " + (myPage + start_page) + " >>>>>>>>>>>>>>>>>>>>>> Conflicting keys for potentially undetected section/article: key = " + p.key());
                                    System.out.println("Existing: '" + p.key() + "' >> " + ret.get(p.key()).text);
                                    System.out.println("New: '" + p.key() + "' >> " + p.text + "\n\n");
                                }
                                ret.put(p.key(), p);
                            }
                        }
                        if (debug) {
                            System.out.println(el.labeledString() + "  <<<<  " + ((el.prevElement != null) ? el.prevElement.labeledString() : ("null:" + el.prevNullReason)));
                        }
                    }
                }
            }

            if (debug) {
                System.out.println("\n-------------------------------------------------------------------------------------------------------------\n");
            }

            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return ret;
    }

    public static void printExtractedParagraphs(Map<String, ArticleParagraph> paragraphs, PrintWriter printer) {
        int secPrev = -1;
        int artPrev = -1;
        int parPrev = -1;
        for (Map.Entry<String, ArticleParagraph> en : paragraphs.entrySet()) {
            if ((en.getValue().sectionNum != secPrev)) {
                artPrev = -1;
                parPrev = -1;
                if (printer != null) {
                    printer.println("\n\nS: ==============================================================================================================");
                }
            }
            if ((en.getValue().articleNum != artPrev)) {
                parPrev = -1;
                if (printer != null) {
                    printer.println("\nA: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                }
            }
            if ((en.getValue().paragraphNum != parPrev)) {
                if (printer != null) {
                    printer.println("P: --------------------------------------------------------------------------------------------------------------");
                }
            }
            if (en.getKey().endsWith("fe")) {
                String lang = languageDetector.detectLanguageOf(en.getValue().text.toLowerCase().trim()).getIsoCode639_1().toString();
                if (printer != null) {
                    printer.println("***************** MISS-LANG ************* P# " + (en.getValue().page + 1) + " '" + en.getKey() + "' (" + lang + "): " + en.getValue().text);
                }
            } else {
                if (printer != null) {
                    printer.println("P# " + (en.getValue().page + 1) + " '" + en.getKey() + "': " + en.getValue().text);
                }
            }
            artPrev = en.getValue().articleNum;
            parPrev = en.getValue().paragraphNum;
            secPrev = en.getValue().sectionNum;
        }
        if (printer != null) {
            printer.println("--------------------------------------------------------------------------------------------------------------");
            printer.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
            printer.println("==============================================================================================================\n\n");
        }
    }

    public static Map<String, Map<String, GazetteParallelDataExtractor.ArticleParagraph>> pruneParagraphs(Map<String, GazetteParallelDataExtractor.ArticleParagraph> pars) {
        Map<String, Map<String, GazetteParallelDataExtractor.ArticleParagraph>> results = new TreeMap<>();
        for (GazetteParallelDataExtractor.ArticleParagraph a : pars.values()) {
            String key = a.commonKey();
            Map<String, GazetteParallelDataExtractor.ArticleParagraph> par = results.get(key);
            if (par == null) {
                par = new HashMap<>();
            }
            par.put(a.language, a);
            results.put(key, par);
        }
        Map<String, Map<String, GazetteParallelDataExtractor.ArticleParagraph>> joinPrev = new TreeMap<>();
        Set<String> remove = new HashSet<>();
        for (Map.Entry<String, Map<String, GazetteParallelDataExtractor.ArticleParagraph>> entry : results.entrySet()) {
            try {
                Map<String, GazetteParallelDataExtractor.ArticleParagraph> curr = entry.getValue();
                String[] kt = entry.getKey().split("_");
                int sec = Integer.parseInt(kt[0].trim());
                int art = Integer.parseInt(kt[1].trim());
                int par = Integer.parseInt(kt[2].trim());
                if (par == 1) {
                    if ((curr.size() < 2) || (!curr.containsKey("rw")) || (curr.containsKey("fe"))) {
                        remove.add(entry.getKey());
                    }
                } else {
                    String headKey = GazetteParallelDataExtractor.commonParagraphKey(sec, art, 1);
                    Map<String, GazetteParallelDataExtractor.ArticleParagraph> head = results.get(headKey);
                    if ((!remove.contains(headKey)) &&
                            (head != null) &&
                            (head.size() > 1) &&
                            (!curr.containsKey("fe"))) {
                        if (curr.size() < head.size()) {
                            for (int prevPar = (par - 1); prevPar > 0; prevPar--) {
                                Map<String, GazetteParallelDataExtractor.ArticleParagraph> prev = results.get(GazetteParallelDataExtractor.commonParagraphKey(sec, art, prevPar));
                                if (prev != null) {
                                    if (prev.size() == head.size()) {
                                        for (String lang : entry.getValue().keySet()) {
                                            if (prev.containsKey(lang)) {
                                                prev.get(lang).text = prev.get(lang).text.trim() + " " + entry.getValue().get(lang).text.trim();
                                            }
                                        }
                                        break;
                                    }
                                } else {
                                    remove.add(entry.getKey());
                                    break;
                                }
                            }
                            remove.add(entry.getKey());
                        }
                    } else {
                        remove.add(entry.getKey());
                    }
                }
            } catch (Exception ex) {
                remove.add(entry.getKey());
            }
        }
        for (String key : remove) {
            results.remove(key);
        }
        return results;
    }

    public static void generateParallelGazetteData(String newGazettesFolder,
                                                   String outputFile) throws IOException {
        File dir = new File(newGazettesFolder);
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile), true);
        int count = 0;
        if (dir.isDirectory()) {
            File[] dirFiles = dir.listFiles();
            if (dirFiles != null) {
                Long startMs = System.currentTimeMillis();
                for (int indx = 0; indx < dirFiles.length; indx++) {
                    ServiceFactory.completeAndBeginHibernateTransaction();
                    System.out.println(">>Processed " + indx + "/" + dirFiles.length + " " + String.format("%.2f", (100.0 * indx / ((double) dirFiles.length))) + "%   " +
                            ETA.printETA(startMs, (double) indx, (double) dirFiles.length));
                    File file = dirFiles[indx];
                    if (file.exists() && file.isAbsolute()) {
                        writer.println((++count) + ". Processing .... " + file.getAbsolutePath());
                        if (GazetteParDoc.DAO().findByUniqueIndex("uri", file.getName()) == null) {
                            try {
                                StringBuilder bld = new StringBuilder();
                                Map<String, ArticleParagraph> pars = extractParagraphs(file.getAbsolutePath());
                                Map<String, Map<String, ArticleParagraph>> prunedData = pruneParagraphs(pars);
                                Map<String, Map<String, String>> content = new TreeMap<>();
                                for (Map.Entry<String, Map<String, ArticleParagraph>> entry : prunedData.entrySet()) {
                                    Map<String, String> vals = new HashMap<>();
                                    for (Map.Entry<String, GazetteParallelDataExtractor.ArticleParagraph> pair : entry.getValue().entrySet()) {
                                        if ("rw".equals(pair.getKey())) {
                                            bld.append(pair.getValue().text.trim()).append("\n");
                                        }
                                        vals.put(pair.getKey(), pair.getValue().text.trim());
                                    }
                                    content.put(entry.getKey(), vals);
                                }
//                                String uri = file.getName();
//                            String docId = System.currentTimeMillis() + "_" + file.getName();
//                            try {
//                                LDoc ld = BFSNodeProcessorThread.saveLDoc(bld.toString(), "gazette", uri);
//                                if (ld != null) {
//                                    docId = ld.getId().toString();
//                                }
//                            } catch (Exception ex) {
//                                System.out.println(ex.toString());
//                            }
                                GazetteParDoc pd = new GazetteParDoc("gazette", file.getName(), false, content).persist();
                                writer.println(count + ". ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~> Saved " + file.getName() + "..................................... " + " | " + (pd != null) + " ==> " + content.size());
                            } catch (Exception ex) {
                                ex.printStackTrace();
                            }
                        } else {
                            writer.println(count + ". ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~> GazetteParDoc " + file.getName() + "..................................... exists!");
                        }
                        System.gc();
                    }
                }
            }
        }
        writer.close();
    }
}
