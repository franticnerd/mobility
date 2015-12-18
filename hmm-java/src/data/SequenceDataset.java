package data;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.util.*;

/**
 * HMM training data.
 * Created by chao on 4/29/15.
 */
public class SequenceDataset {

    // training data
    List<Sequence> trainseqs;
    List<Map<Integer, Integer>> textData;  // The text data for the R trainseqs, the length is 2R
    List<RealVector> geoData;  // The geographical data for the R seqeunces, length 2R
    List<RealVector> temporalData;  // The temporal data for the R seqeunces, length 2R
    int numWords;
    // test data
    double testRatio;
    List<Sequence> testSeqs;

    public void load(String sequenceFile) throws IOException {
        testRatio = 0;
        load(sequenceFile, 0);
    }

    public void load(String sequenceFile, double testRatio) throws IOException {
        this.testRatio = testRatio;
        List<Sequence> allSeqs = new ArrayList<Sequence>();
        BufferedReader br = new BufferedReader(new FileReader(sequenceFile));
        while(true) {
            String line = br.readLine();
            if(line == null) break;
            Sequence seq = parseSequence(line);
            allSeqs.add(seq);
        }
        br.close();
        trainseqs = allSeqs.subList(0, (int)(allSeqs.size() * ( 1- testRatio)));
        testSeqs = allSeqs.subList((int)(allSeqs.size() * ( 1- testRatio)), allSeqs.size());
        // Geo, temporal and text data.
        geoData = new ArrayList<RealVector>();
        temporalData = new ArrayList<RealVector>();
        textData = new ArrayList<Map<Integer, Integer>>();
        for (Sequence sequence : trainseqs) {
            if (sequence.size() != 2) {
                System.out.println("Warning! The sequence's length is not 2.");
            }
            List<Checkin> checkins = sequence.getCheckins();
            for (Checkin c : checkins) {
                geoData.add(c.getLocation().toRealVector());
                textData.add(c.getMessage());
                temporalData.add(new ArrayRealVector(new double[] {c.getTimestamp() % 1440})); // get the minutes of the timestamp.
            }
        }
        System.out.println("Loading geo, temporal, and textual data finished.");
    }

    public void setNumWords(int numWords) {
        this.numWords = numWords;
    }

    // Each line contains: checkin Id, userId, placeid, timestamp, message
    private Sequence parseSequence(String line) {
        String[] items = line.split(",");
        Checkin start = toCheckin(Arrays.copyOfRange(items, 0, items.length / 2));
        Checkin end = toCheckin(Arrays.copyOfRange(items, items.length / 2, items.length));
        long userId = start.getUserId();
        Sequence seq = new Sequence(userId);
        seq.addCheckin(start);
        seq.addCheckin(end);
        return seq;
    }

    private Checkin toCheckin(String[] items) {
        if(items.length < 6) {
            System.out.println("Error when parsing checkins.");
            return null;
        }
        int checkinId = Integer.parseInt(items[0]);
        int timestamp = Integer.parseInt(items[1]);
        long userId = Long.parseLong(items[2]);
        double lat = Double.parseDouble(items[3]);
        double lng = Double.parseDouble(items[4]);
        Map<Integer, Integer> message = parseMessage(items[5]);
        return new Checkin(checkinId, timestamp, userId, lat, lng, message);
    }

    private Map<Integer, Integer> parseMessage(String s) {
        Map<Integer, Integer> message = new HashMap<Integer, Integer>();
        String [] items = s.split("\\s");
        if(items.length == 0) {
            System.out.println("Warning! Checkin has no message.");
        }
        for (int i = 0; i < items.length; i++) {
            int wordId = Integer.parseInt(items[i]);
            message.put(wordId, 1);
        }
        return message;
    }

    public List<Sequence> getSequences() {
        return trainseqs;
    }

    public List<RealVector> getGeoData() {
        return geoData;
    }

    public List<RealVector> getTemporalData() {
        return temporalData;
    }

    public List<Map<Integer, Integer>> getTextData() {
        return textData;
    }

    public RealVector getGeoDatum(int index) {
        return geoData.get(index);
    }

    public RealVector getTemporalDatum(int index) {
        return temporalData.get(index);
    }

    public Map<Integer, Integer> getTextDatum(int index) {
        return textData.get(index);
    }

    public Sequence getSequence(int i) {
        return trainseqs.get(i);
    }

    public int size() {
        return trainseqs.size();
    }

    public int numWords() {
        return numWords;
    }

    public PredictionDataset extractTestData() throws Exception {
        return new PredictionDataset(testSeqs);
    }

}
