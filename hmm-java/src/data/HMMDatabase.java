package data;

import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * HMM training data.
 * Created by chao on 4/29/15.
 */
public class HMMDatabase {

    List<Sequence> sequences;
    List<Map<Integer, Integer>> textData;  // The text data for the R sequences, the length is 2R
    List<RealVector> geoData;  // The geographical data for the R seqeunces, length 2R
    int numWords;

    public void load(int minGap, int maxGap) {
        // Extract sequences.
        sequences = Database.sd.getDenseSequences(minGap, maxGap);
        System.out.println("Finished extracting dense sequences. Count:" + sequences.size());
        // Number of words.
        numWords = Database.wd.size();
        // Geo and text data.
        geoData = new ArrayList<RealVector>();
        textData = new ArrayList<Map<Integer, Integer>>();
        for (Sequence sequence : sequences) {
            if (sequence.size() != 2) {
                System.out.println("Warning! The sequence's length is not 2.");
            }
            List<Checkin> checkins = sequence.getCheckins();
            for (Checkin c : checkins) {
                Place p = Database.pd.getPlace(c.getPlaceId()).copy();
                p.addMessage(c.getMessage());
                geoData.add(p.getLocation().toRealVector());
                textData.add(p.getDescriptions());
            }
        }
        System.out.println("Loading geo and textual data finished.");
    }

    public List<Sequence> getSequences() {
        return sequences;
    }

    public List<RealVector> getGeoData() {
        return geoData;
    }

    public List<Map<Integer, Integer>> getTextData() {
        return textData;
    }

    public RealVector getGeoDatum(int index) {
        return geoData.get(index);
    }

    public Map<Integer, Integer> getTextDatum(int index) {
        return textData.get(index);
    }
    public int numSequences() {
        return sequences.size();
    }

    public int numWords() {
        return numWords;
    }



    // Output the test sequences.
    public void writeTestSequences(String outputFile, int size) throws Exception {
        List<Sequence> testSeqs = extractTestSequences(size);
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile, false));
        for (Sequence seq : testSeqs)
            bw.append(seq.toStringForLengthTwo() + "\n");
        bw.close();
    }

//    public List<Sequence> extractTestSequences(int size) throws Exception {
//        List<Sequence> testSeqs = new ArrayList<Sequence>();
//        int[] indices = ArrayUtils.genKRandomNumbers(sequences.size(), size);
//        for (int i = 0; i < indices.length; i++) {
//            int index = indices[i];
//            Sequence s = sequences.get(index);
//            testSeqs.add(s);
//        }
//        return testSeqs;
//    }

//    public List<Sequence> extractTestSequences(int size) throws Exception {
//        List<Sequence> testSeqs = new ArrayList<Sequence>();
//        for (int i = 0; i < size; i++) {
//            int index = sequences.size() - size + i;
//            Sequence s = sequences.get(index);
//            testSeqs.add(s);
//        }
//        return testSeqs;
//    }

    public List<Sequence> extractTestSequences(int size) throws Exception {
        List<Sequence> testSeqs = new ArrayList<Sequence>();
        for (Sequence s : sequences) {
            if (s.getCheckin(0).getPlaceId() == 3188)
                testSeqs.add(s);
        }
        return testSeqs;
    }

}
