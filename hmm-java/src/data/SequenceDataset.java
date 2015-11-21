package data;

import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.util.*;

/**
 * HMM training data.
 * Created by chao on 4/29/15.
 */
public class SequenceDataset {

    List<Sequence> sequences;
    List<Map<Integer, Integer>> textData;  // The text data for the R sequences, the length is 2R
    List<RealVector> geoData;  // The geographical data for the R seqeunces, length 2R
    int numWords;

    public void load(String sequenceFile) throws IOException {
        sequences = new ArrayList<Sequence>();
        BufferedReader br = new BufferedReader(new FileReader(sequenceFile));
        while(true) {
            String line = br.readLine();
            if(line == null) break;
            Sequence seq = parseSequence(line);
            sequences.add(seq);
        }
        br.close();
        // Geo and text data.
        geoData = new ArrayList<RealVector>();
        textData = new ArrayList<Map<Integer, Integer>>();
        for (Sequence sequence : sequences) {
            if (sequence.size() != 2) {
                System.out.println("Warning! The sequence's length is not 2.");
            }
            List<Checkin> checkins = sequence.getCheckins();
            for (Checkin c : checkins) {
                geoData.add(c.getLocation().toRealVector());
                textData.add(c.getMessage());
            }
        }
        System.out.println("Loading geo and textual data finished.");
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

    public Sequence getSequence(int i) {
        return sequences.get(i);
    }

    public int size() {
        return sequences.size();
    }

    public int numWords() {
        return numWords;
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
        return testSeqs.subList((int)(sequences.size() * 0.8), sequences.size());
    }

    // get the candidate location for the target sequence
    public List<Checkin> getCandidate(int i) {
        List<Checkin> ret = new ArrayList<Checkin>();
        for(Sequence s : sequences) {
            ret.add(s.getCheckin(1));
        }
        return ret;
    }

}
