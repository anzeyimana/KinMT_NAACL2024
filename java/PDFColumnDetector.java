
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.text.LegacyPDFStreamEngine;
import org.apache.pdfbox.text.TextPosition;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;

public class PDFColumnDetector extends LegacyPDFStreamEngine {
    List<TextPosition> text = new ArrayList<>();
    Float minX = 999999.0f;
    Float minY = 999999.0f;
    Float maxX = 0.0f;
    Float maxY = 0.0f;

    public PDFColumnDetector() throws IOException {
    }

    public void reset() {
        this.text.clear();
        this.minX = 999999.0f;
        this.minY = 999999.0f;
        this.maxX = 0.0f;
        this.maxY = 0.0f;
    }

    @Override
    public void processPage(PDPage page) throws IOException {
        reset();
        super.processPage(page);
    }

    @Override
    protected void processTextPosition(TextPosition t) {
        super.processTextPosition(t);
        text.add(t);
        if (t.getX() < minX) {
            minX = t.getX();
        }
        if (t.getY() < minY) {
            minY = t.getY();
        }
        if (t.getEndX() > maxX) {
            maxX = t.getEndX();
        }
        if (t.getEndY() > maxY) {
            maxY = t.getEndY();
        }
    }

    private static class Cell1D {
        Float x;
        Float ex;
        Float cx;
        Integer count;
        Integer leftCumCount;
        Integer rightCumCount;

        Integer idx;
        Integer numCells;

        Integer leftDiff;
        Integer rightDiff;

        public Cell1D(float x, float ex, int idx, int numCells) {
            this.x = x;
            this.ex = ex;
            this.cx = x + ((ex - x) / 2.0f);
            this.count = 0;
            this.leftCumCount = 0;
            this.rightCumCount = 0;
            this.leftDiff = 0;
            this.rightDiff = 0;
            this.idx = idx;
            this.numCells = numCells;
        }

        public Float rankLeftDiff() {
            return leftDiff / (((float) count) + 0.1f);
        }

        public Float rankRightDiff() {
            return rightDiff / (((float) count) + 0.1f);
        }

        public int compareRightDiff(Cell1D other, float countMean, float countStd, float leftDiffMean, float leftDiffStd, float rightDiffMean, float rightDiffStd, int maxCols) {
            boolean IAmCool = meetsBorderConditionsByRight(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, maxCols);
            boolean HeIsCool = other.meetsBorderConditionsByRight(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, maxCols);
            if (IAmCool && (!HeIsCool)) {
                return Integer.compare(0, 1);
            } else if ((!IAmCool) && HeIsCool) {
                return Integer.compare(1, 0);
            }
            return Float.compare(other.rankRightDiff(), rankRightDiff());
        }

        public int compareLeftDiff(Cell1D other, float countMean, float countStd, float leftDiffMean, float leftDiffStd, float rightDiffMean, float rightDiffStd, int maxCols) {
            boolean IAmCool = meetsBorderConditionsByLeft(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, maxCols);
            boolean HeIsCool = other.meetsBorderConditionsByLeft(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, maxCols);
            if (IAmCool && (!HeIsCool)) {
                return Integer.compare(0, 1);
            } else if ((!IAmCool) && HeIsCool) {
                return Integer.compare(1, 0);
            }
            return Float.compare(other.rankLeftDiff(), rankLeftDiff());
        }

        public boolean meetsBorderConditionsByRight(float countMean, float countStd, float leftDiffMean, float leftDiffStd, float rightDiffMean, float rightDiffStd, int expectedColumns) {
            if (true) {
                //return true;
            }
            if ((countMean - countStd) < 2.0f) {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) rightDiff) >= (rightDiffMean + (1.5f * rightDiffStd))))
                        && (((float) rightCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            } else {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) rightDiff) >= (rightDiffMean + (1.5f * rightDiffStd))))
                        && (((float) rightCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            }
        }

        public boolean meetsBorderConditionsByLeft(float countMean, float countStd, float leftDiffMean, float leftDiffStd, float rightDiffMean, float rightDiffStd, int expectedColumns) {
            if (true) {
                //return true;
            }
            if ((countMean - countStd) < 2.0f) {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) leftDiff) >= (leftDiffMean + (1.0f * leftDiffStd))))
                        && (((float) leftCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            } else {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) leftDiff) >= (leftDiffMean + (1.0f * leftDiffStd))))
                        && (((float) leftCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            }
        }

        public boolean meetsBorderConditions(float countMean, float countStd, float leftDiffMean, float leftDiffStd, float rightDiffMean, float rightDiffStd, int expectedColumns) {
            if (true) {
                //return true;
            }
            if ((countMean - countStd) < 2.0f) {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) rightDiff) >= (rightDiffMean + (1.5f * rightDiffStd))) ||
                        (((float) leftDiff) >= (leftDiffMean + (1.0f * leftDiffStd))))
                        && (((float) leftCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))))
                        && (((float) rightCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            } else {
                return (((((float) count) < (countMean - (1.0f * countStd))) && (count < 5)) || (count == 0) || (((countMean - countStd) < 2.0f) && (count < 4)))
                        && ((((float) rightDiff) >= (rightDiffMean + (1.5f * rightDiffStd))) ||
                        (((float) leftDiff) >= (leftDiffMean + (1.0f * leftDiffStd))))
                        && (((float) leftCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))))
                        && (((float) rightCumCount) >= (((float) (leftCumCount + rightCumCount)) / ((float) 16.0f * (expectedColumns + 1))));
            }
        }
    }

    public Float countRequiredPercentDiff(List<Cell1D> cells, int expectedCols, Float x, Float y) {
        float l = (x <= y) ? x : y;
        float r = (x <= y) ? y : x;
        Cell1D lc = cells.get(1);
        for (Cell1D c : cells) {
            if (l <= c.ex) {
                lc = c;
                break;
            }
        }
        Cell1D rc = cells.get(cells.size() - 1);
        for (Cell1D c : cells) {
            if (r <= c.ex) {
                rc = c;
                break;
            }
        }
        return ((float) (lc.rightCumCount - rc.rightCumCount)) / (((float) (lc.leftCumCount + lc.rightCumCount)) / ((float) (expectedCols + 1.0f)));
    }

    public List<Float> getColumnSeparators(int expectedCols, boolean debug) {
        float avgFontSize = 0.0f;
        float minFontSize = 50000.0f;
        for (TextPosition p : text) {
            avgFontSize += p.getFontSize();
            if (p.getFontSize() < minFontSize) {
                if (p.getFontSize() > 0.0) {
                    minFontSize = p.getFontSize();
                }
            }
        }
        if (avgFontSize > 0.0f) {
            avgFontSize = avgFontSize / ((float) text.size());
        }
        if (minFontSize >= 50000.0f) {
            minFontSize = 4.0f;
        }
        float d = minFontSize / 4.0f;
        //d = 1.0f;
        if (debug) {
            System.out.println("minFontSize: " + minFontSize);
            System.out.println("avgFontSize: " + avgFontSize);
            System.out.println("d: " + d);
            System.out.println("minX: " + minX);
            System.out.println("maxX: " + maxX);
        }
        float x = minX;
        float ex = x + d;
        List<Cell1D> cells = new ArrayList<>();
        while (ex <= maxX) {
            cells.add(new Cell1D(x, ex, cells.size(), 0));
            x = ex;
            ex = x + d;
        }
        for (Cell1D c : cells) {
            c.numCells = cells.size();
        }
        if (debug) {
            System.out.println("Got " + cells.size() + " cells.");
        }
        for (Cell1D cell : cells) {
            for (TextPosition p : text) {
                if ((p.getX() <= cell.cx) && (p.getEndX() >= cell.cx)) {
                    cell.count += 1;
                }
            }
        }
        int left = 0;
        for (int i = 0; i < cells.size(); i++) {
            if (i > 0) {
                cells.get(i).leftDiff = Math.abs(cells.get(i).count - cells.get(i - 1).count);
            } else {
                cells.get(i).leftDiff = 0;
            }
            cells.get(i).leftCumCount = left;
            left += cells.get(i).count;
        }
        int right = 0;
        for (int i = (cells.size() - 1); i >= 0; i--) {
            if (i < (cells.size() - 1)) {
                cells.get(i).rightDiff = Math.abs(cells.get(i).count - cells.get(i + 1).count);
            } else {
                cells.get(i).rightDiff = 0;
            }
            cells.get(i).rightCumCount = right;
            right += cells.get(i).count;
        }
        float leftSum = 0.0f;
        float leftSumSq = 0.0f;
        float rightSum = 0.0f;
        float rightSumSq = 0.0f;
        float countSum = 0.0f;
        float countSumSq = 0.0f;

        float leftDiffSum = 0.0f;
        float leftDiffSumSq = 0.0f;

        float rightDiffSum = 0.0f;
        float rightDiffSumSq = 0.0f;
        for (Cell1D c : cells) {
            countSum += (float) (c.count);
            countSumSq += (float) (c.count * c.count);

            leftSum += (float) (c.leftCumCount);
            leftSumSq += (float) (c.leftCumCount * c.leftCumCount);

            rightSum += (float) (c.rightCumCount);
            rightSumSq += (float) (c.rightCumCount * c.rightCumCount);

            leftDiffSum += (float) (c.leftDiff);
            leftDiffSumSq += (float) (c.leftDiff * c.leftDiff);

            rightDiffSum += (float) (c.rightDiff);
            rightDiffSumSq += (float) (c.rightDiff * c.rightDiff);
        }

        final float leftMean = leftSum / ((float) cells.size());
        final float rightMean = rightSum / ((float) cells.size());
        final float countMean = countSum / ((float) cells.size());
        final float leftDiffMean = leftDiffSum / ((float) cells.size());
        final float rightDiffMean = rightDiffSum / ((float) cells.size());

        final float leftStd = (float) Math.sqrt((leftSumSq / ((float) cells.size())) - (leftMean * leftMean));
        final float rightStd = (float) Math.sqrt((rightSumSq / ((float) cells.size())) - (rightMean * rightMean));

        final float countStd = (float) Math.sqrt((countSumSq / ((float) cells.size())) - (countMean * countMean));

        final float leftDiffStd = (float) Math.sqrt((leftDiffSumSq / ((float) cells.size())) - (leftDiffMean * leftDiffMean));
        final float rightDiffStd = (float) Math.sqrt((rightDiffSumSq / ((float) cells.size())) - (rightDiffMean * rightDiffMean));

        List<Cell1D> leftBorders = new ArrayList<>();
        cells.sort((o1, o2) -> o1.compareLeftDiff(o2, countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, expectedCols));
        for (int i = 0; (i < (4 * expectedCols)) && (i < cells.size()); i++) {
            leftBorders.add(cells.get(i));
        }

        List<Cell1D> rightBorders = new ArrayList<>();
        cells.sort((o1, o2) -> o1.compareRightDiff(o2, countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, expectedCols));
        for (int i = 0; (i < (4 * expectedCols)) && (i < cells.size()); i++) {
            rightBorders.add(cells.get(i));
        }
        if (debug) {
            System.out.println("Left: Avg: " + leftMean + " Stdev: " + leftStd);
            System.out.println("Right: Avg: " + rightMean + " Stdev: " + rightStd);
            System.out.println("Count: Avg: " + countMean + " Stdev: " + countStd);

            System.out.println("Left-Diff: Avg: " + leftDiffMean + " Stdev: " + leftDiffStd);
        }
        List<Cell1D> myLeft = new ArrayList<>();
        List<Cell1D> myRight = new ArrayList<>();
        Cell1D leftPrev = null;
        for (Cell1D c : leftBorders) {
            if (c.meetsBorderConditions(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, expectedCols)) {
                if (leftPrev != null) {
                    if (
                        //((((float) (leftPrev.leftDiff - c.leftDiff)) > (1.0f * leftDiffStd)) && (c.count > 3)) ||
                            (Math.abs(leftPrev.cx - c.cx) < (Math.abs(maxX - minX) / ((float) ((expectedCols + 1) * 4)))) ||
                                    (((float) (c.count - leftPrev.count)) > (1.0f * countStd))) {
                        if (debug) {
                            System.out.println("\n\t****> Cell(" + c.idx + "/" + c.numCells + " " + c.rankLeftDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                        }
                        break;
                        //System.out.println("Conflict: ");
                    }
                }
                myLeft.add(c);
                if (debug) {
                    System.out.println("\n\t>>>>> Cell(" + c.idx + "/" + c.numCells + " " + c.rankLeftDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                }
                leftPrev = c;
            } else {
                if (debug) {
                    System.out.println("\n\t--> Cell(" + c.idx + "/" + c.numCells + " " + c.rankLeftDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                }
            }
        }

        if (debug) {
            System.out.println("Right-Diff: Avg: " + rightDiffMean + " Stdev: " + rightDiffStd);
        }
        Cell1D rightPrev = null;
        for (Cell1D c : rightBorders) {
            if (c.meetsBorderConditions(countMean, countStd, leftDiffMean, leftDiffStd, rightDiffMean, rightDiffStd, expectedCols)) {
                if (rightPrev != null) {
                    if (
                        //((((float) (rightPrev.rightDiff - c.rightDiff)) > (1.0f * rightDiffStd)) && (c.count > 3)) ||
                            (Math.abs(rightPrev.cx - c.cx) < (Math.abs(maxX - minX) / ((float) ((expectedCols + 1) * 4)))) ||
                                    (((float) (c.count - rightPrev.count)) > (1.0f * countStd))) {
                        if (debug) {
                            System.out.println("\n\t****> Cell(" + c.idx + "/" + c.numCells + " " + c.rankRightDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                        }
                        break;
                    }
                }
                myRight.add(c);
                if (debug) {
                    System.out.println("\n\t>>>>> Cell(" + c.idx + "/" + c.numCells + " " + c.rankRightDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                }
                rightPrev = c;
            } else {
                if (debug) {
                    System.out.println("\n\t--> Cell(" + c.idx + "/" + c.numCells + " " + c.rankRightDiff() + "):" + " LD: " + c.leftDiff + " RD: " + c.rightDiff + " CNT: " + c.count + " LFT: " + c.leftCumCount + " RGT: " + c.rightCumCount + " <" + c.cx + ">\n");
                }
            }
        }
        myLeft.sort((o1, o2) -> Float.compare(o1.cx, o2.cx));
        myRight.sort((o1, o2) -> Float.compare(o2.cx, o1.cx));
        Set<Integer> used = new HashSet<>();
        Map<Float, Float> result = new HashMap<>();
        for (int s = 0; s < myLeft.size(); s++) {
            for (int t = 0; t < myRight.size(); t++) {
                Cell1D l = myLeft.get(s);
                Cell1D r = myRight.get(t);
                boolean lOK = true;
                boolean rOK = true;
                if (s > 0) {
                    Cell1D pl = myLeft.get(s - 1);
//                    if (debug) {
//                        System.out.println("\n\nEvaluating l#" + l.idx +" ~ p#"+pl.idx);
//                        System.out.println("@@@@@@@@@@@@@ l.leftDiff: " + (l.leftDiff));
//                        System.out.println("@@@@@@@@@@@@@ (((float) pl.leftDiff) / 2.0f): " + ((((float) pl.leftDiff) / 2.0f)));
//                        System.out.println("@@@@@@@@@@@@@ l.leftCumCount - pl.leftCumCount: " + (l.leftCumCount - pl.leftCumCount));
//                        System.out.println("@@@@@@@@@@@@@@ (((float) (l.leftCumCount + l.rightCumCount)) / ((float) ((expectedCols + 1) * 4))): " + (((float) (l.leftCumCount + l.rightCumCount)) / ((float) ((expectedCols + 1) * 4))));
//                        System.out.println("\n\n");
//                    }
                    if (((float) l.leftDiff) < (((float) pl.leftDiff) / 2.0f)) {
                        if ((l.leftCumCount - pl.leftCumCount) < (((float) (l.leftCumCount + l.rightCumCount)) / ((float) ((expectedCols + 1) * 4)))) {
                            if (debug) {
                                System.out.println("=========> L-Crushing # " + l.idx);
                            }
                            lOK = false;
                        }
                    }
                }
                if (t > 0) {
                    Cell1D pr = myRight.get(t - 1);
                    if (((float) r.rightDiff) < (((float) pr.rightDiff) / 2.0f)) {
                        if ((r.rightCumCount - pr.rightCumCount) < (((float) (r.leftCumCount + r.rightCumCount)) / ((float) ((expectedCols + 1) * 4)))) {
                            rOK = false;
                            if (debug) {
                                System.out.println("=========> R-Crushing # " + l.idx);
                            }
                        }
                    }
                }
                if (lOK && rOK && (l != r) && (!l.cx.equals(r.cx))) {
                    float x1 = (((float) l.rightCumCount) / ((float) (l.leftCumCount + l.rightCumCount))) - (((float) r.rightCumCount) / ((float) (r.leftCumCount + r.rightCumCount)));
                    float x2 = (((float) l.leftCumCount) / ((float) (l.leftCumCount + l.leftCumCount))) - (((float) r.leftCumCount) / ((float) (r.leftCumCount + r.leftCumCount)));
                    float x3 = (r.cx - l.cx) / (maxX - minX);
                    float x4 = ((float) (l.count - r.count)) / ((float) (l.count + r.count));
                    float val = (float) Math.sqrt((x1 * x1) + (x2 * x2) + (x3 * x3) + (x4 * x4));
                    if ((l.cx < r.cx) && ((maxX - minX) > 0.0f)) {
                        if ((r.cx - l.cx) < ((maxX - minX) / ((((float) expectedCols) + 1.0f) * 4.0f))) {
                            used.add(l.idx);
                            used.add(r.idx);

                            float compVal = x3 * x3;

                            if (debug) {
                                System.out.println("Associating: l# " + l.idx + " and r# " + r.idx + " -----> diff: " + x3);
                            }

                            if (l.count == 0) {
                                result.put(l.cx, compVal);
                                if (debug) {
                                    // System.out.println("Left.......");
                                }
                            } else if (r.count == 0) {
                                result.put(r.cx, compVal);
                                if (debug) {
                                    // System.out.println("Right.......");
                                }
                            } else {
                                result.put(l.cx + ((r.cx - l.cx) / 2.0f), compVal);
                                if (debug) {
                                    // System.out.println("Avg.......");
                                }
                            }

                            if (debug) {
                                // System.out.println(l.cx + ", " + r.cx + " >> " + (x3 * x3));
                            }
                        } else {
                            if (debug) {
                                System.out.println("=========> @1: Avoiding l# " + l.idx + " x r#" + r.idx);
                            }
                        }
                    } else {
                        if (debug) {
                            System.out.println("=========> @@@@2: Avoiding l# " + l.idx + " x r#" + r.idx);
                        }
                    }
                }
            }
        }

        if (result.size() < expectedCols) {
            for (Cell1D l : myLeft) {
                for (Cell1D r : myRight) {
                    if (((l == r) || (l.cx.equals(r.cx))) && (!used.contains(l.idx))) {
                        used.add(r.idx);
                        result.put(l.cx, 0.0f);
                        if (result.size() >= expectedCols) {
                            break;
                        }
                    }
                }
            }
        }

        for (Cell1D l : myLeft) {
            if ((!used.contains(l.idx)) && (l.count == 0)) {
                result.put(l.cx, 0.0f);
                used.add(l.idx);
            }
        }
        for (Cell1D r : myRight) {
            if ((!used.contains(r.idx)) && (r.count == 0)) {
                result.put(r.cx, 0.0f);
                used.add(r.idx);
            }
        }

        // Sort by distance to expected column lines
        float colWidth = ((float) (maxX - minX)) / ((float) (expectedCols + 1.0));
        for (Float k : result.keySet()) {
            float min = 2.0f * maxX;
            for (int i = 1; i <= expectedCols; i++) {
                float sample = ((float) minX) + ((float) (colWidth * i));
                float t = Math.abs(k - sample);
                if (t < min) {
                    min = t;
                }
            }
            if (debug) {
                System.out.println("$$$$$$$$ New order: k# " + k + " >>> #v " + min);
            }
            result.put(k, min);
        }

        Map<Float, Float> sortedResult = MapValueSorter.sortByValue(result, true);

        List<Float> resList = new ArrayList<>();
        int lim = 0;
        List<Map.Entry<Float, Float>> entries = new ArrayList<>(sortedResult.entrySet());
        for (int i = 0; i < entries.size(); i++) {
            boolean keep = true;
            Map.Entry<Float, Float> ent = entries.get(i);
            if ((i > 0) && (i < (entries.size() - 1))) {
                Map.Entry<Float, Float> prev = entries.get(i - 1);
                Map.Entry<Float, Float> next = entries.get(i + 1);
                if (Math.abs(ent.getKey() - prev.getKey()) < (colWidth / 4.0f)) {
                    float prevDiff = countRequiredPercentDiff(cells, expectedCols, ent.getKey(), prev.getKey());
                    float nextDiff = countRequiredPercentDiff(cells, expectedCols, ent.getKey(), next.getKey());
                    if (debug) {
                        System.out.println("PrevDiff: " + prevDiff);
                        System.out.println("nextDiff: " + nextDiff);
                        System.out.println("Math.abs(ent.getKey() - prev.getKey()): " + Math.abs(ent.getKey() - prev.getKey()));
                        System.out.println("(colWidth / 4.0f): " + (colWidth / 4.0f));
                    }
                    if ((prevDiff < 0.5f) && (nextDiff > 0.7f)) {
                        keep = false;
                    }
                }
            }
            if (keep) {
                resList.add(ent.getKey());
                lim++;
                if (lim >= expectedCols) {
                    break;
                }
            }
        }

//        resList.clear();
//        for (Cell1D c : myLeft) {
//            resList.add(c.cx);
//        }

        if ((resList.size() < expectedCols) && (!myRight.isEmpty())) {
            int j = 0;
            for (Cell1D c : myRight) {
                if ((c.count == 0) && (!used.contains(c.idx))) {
                    used.add(c.idx);
                    resList.add(c.x);
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        if ((resList.size() < expectedCols) && (!myLeft.isEmpty())) {
            int j = 0;
            for (Cell1D c : myLeft) {
                if ((c.count == 0) && (!used.contains(c.idx))) {
                    used.add(c.idx);
                    resList.add(c.ex);
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        if ((resList.size() < expectedCols) && (!myRight.isEmpty())) {
            int j = 0;
            for (Cell1D c : myRight) {
                if ((c.count < 4) && (!used.contains(c.idx))) {
                    used.add(c.idx);
                    resList.add(c.x);
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        if ((resList.size() < expectedCols) && (!myLeft.isEmpty())) {
            int j = 0;
            for (Cell1D c : myLeft) {
                if ((c.count < 4) && (!used.contains(c.idx))) {
                    used.add(c.idx);
                    resList.add(c.ex);
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        if (resList.isEmpty() && (!myRight.isEmpty())) {
            int j = 0;
            for (Cell1D c : myRight) {
                if ((((c.count < 4) && (c.rightDiff > 5)) || (c.rightDiff > (2 * c.count))) && (!used.contains(c.idx))) {
                    resList.add(c.x);
                    used.add(c.idx);
                    if (debug) {
                        System.out.println("+++ Inconveniently adding:# " + c.idx);
                    }
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        if (resList.isEmpty() && (!rightBorders.isEmpty())) {
            int j = 0;
            for (Cell1D c : rightBorders) {
                if ((((c.count < 4) && (c.rightDiff > 5)) || ((c.rightDiff > 12) && (c.rightDiff > (2 * c.count))))
                        && (!used.contains(c.idx))
                        && (c.leftCumCount > ((float) (c.leftCumCount + c.rightCumCount)) / ((float) (4 * (expectedCols + 1))))) {
                    resList.add(c.x);
                    used.add(c.idx);
                    if (debug) {
                        System.out.println("~~~ Inconveniently adding:# " + c.idx);
                    }
                    j++;
                    if (j >= expectedCols) {
                        break;
                    }
                }
            }
        }
        resList.sort(Float::compare);
        return resList;
    }

    public static void testColumns(String pdfFile, int page, int expectedCols, boolean debug) {
        try {
            PDDocument doc = PDDocument.load(new File(pdfFile));
            PDFColumnDetector det = new PDFColumnDetector();
            det.processPage(doc.getPage(page));
            System.out.println(LocalDateTime.now() + " > STARTED: PAGE: " + doc.getPage(page).getCropBox().getWidth());
            det.getColumnSeparators(expectedCols, debug);
            System.out.println(LocalDateTime.now() + " > ENDED");
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }


    public static void markDocumentColumns(String inputFile, String outputFile, int expectedCols, boolean debug) {
        try {
            PDDocument doc = PDDocument.load(new File(inputFile));
            int p = 0;
            for (PDPage page : doc.getPages()) {
                PDFColumnDetector det = new PDFColumnDetector();
                det.processPage(page);
                List<Float> cols = det.getColumnSeparators(expectedCols, debug);
                int c = 0;
                for (Float f : cols) {
                    PDPageContentStream stream = new PDPageContentStream(doc, page, PDPageContentStream.AppendMode.APPEND, true);
                    stream.setStrokingColor(80, 254, 80);
                    stream.moveTo(f, 0.0f);
                    stream.lineTo(f, page.getCropBox().getHeight());
                    stream.stroke();
                    stream.close();
                    c++;
                    if (c >= expectedCols) {
                        break;
                    }
                }
                p++;
                if (debug) {
                    System.out.println("Page: " + p + " ~~~> " + cols.size());
                }
            }
            doc.save(outputFile);
            doc.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // 97, 229, 297, 281, 1058
//        testSave("Downloads/Official_Gazette_no_05_of_01.02.2010.pdf", 26, 2, true);
//        markDocumentColumns("Habumuremyi.pdf",
//                "Desktop/Habumuremyi_marked_columns.pdf",
//                1, false);
        markDocumentColumns("Desktop/OG_Samples/20090525.pdf",
                "Desktop/20090525_marked.pdf",
                2, false);

//        markDocumentColumns("Downloads/C_OG_no_Special_of_19-03-2020_Amateka_MIFOTRA___RCS___Returning_RNP___Imiryango___Amazina___BNR.pdf",
//                "Desktop/Gazette_marked.pdf",
//                2, false);

//        markDocumentColumns("Downloads/33_Official+Gazette+n%C2%B0+33+of+26-08-2019.pdf",
//                "Downloads/33_Official+Gazette+n%C2%B0+33+of+26-08-2019_marked.pdf",
//                2, false);

//        markDocumentColumns("Downloads/C_OG_N___Special_of_18-07-2019.pdf",
//                "Downloads/C_OG_N___Special_of_18-07-2019_marked.pdf",
//                2,false);
    }
}
