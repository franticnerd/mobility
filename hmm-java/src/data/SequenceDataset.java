package data;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import textAugmentation.Augmenter;

import java.io.*;
import java.util.*;

/**
 * HMM training data. Created by chao on 4/29/15.
 */
public class SequenceDataset {

	// training data
	List<Sequence> trainseqs = new ArrayList<Sequence>();
	List<Map<Integer, Integer>> textData = new ArrayList<Map<Integer, Integer>>(); // The text data for the R trainseqs, the length is 2R
	List<RealVector> geoData = new ArrayList<RealVector>(); // The geographical data for the R seqeunces, length 2R
	List<RealVector> temporalData = new ArrayList<RealVector>(); // The temporal data for the R seqeunces, length 2R
	int numWords;
	// test data
	double testRatio;
	List<Sequence> testSeqs = new ArrayList<Sequence>();

	public void load(String sequenceFile) throws IOException {
		testRatio = 0;
		load(sequenceFile, 0, false);
	}

	public void load(String sequenceFile, double testRatio, boolean filterTest) throws IOException {
		this.testRatio = testRatio;
		List<Sequence> allSeqs = new ArrayList<Sequence>();
		BufferedReader br = new BufferedReader(new FileReader(sequenceFile));
		while (true) {
			String line = br.readLine();
			if (line == null)
				break;
			Sequence seq = parseSequence(line);
			allSeqs.add(seq);
		}
		br.close();

		Collections.shuffle(allSeqs, new Random(1));
		trainseqs = allSeqs.subList(0, (int) (allSeqs.size() * (1 - testRatio)));
		testSeqs = allSeqs.subList((int) (allSeqs.size() * (1 - testRatio)), allSeqs.size());
		if (filterTest) {
			filterTestSeqs(allSeqs);
		}

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
				temporalData.add(new ArrayRealVector(new double[] { c.getTimestamp() % 1440 })); // get the minutes of the timestamp.
			}
		}
		System.out.println("Loading geo, temporal, and textual data finished.");
	}

	public void augmentText(Augmenter augmenter, int augmentedSize, boolean augmentTrain, boolean augmentTest) {
		if (augmentTrain) {
			textData.clear();
			for (Sequence sequence : trainseqs) {
				List<Checkin> checkins = sequence.getCheckins();
				for (Checkin c : checkins) {
					c.setMessage(augmenter.getAugmentedText(c.getMessage(), augmentedSize));
					textData.add(c.getMessage());
				}
			}
		}
		if (augmentTest) {
			for (Sequence sequence : testSeqs) {
				List<Checkin> checkins = sequence.getCheckins();
				for (Checkin c : checkins) {
					c.setMessage(augmenter.getAugmentedText(c.getMessage(), augmentedSize));
				}
			}
		}
	}

	private void filterTestSeqs(List<Sequence> allSeqs) {
		HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();
		for (int i = 0; i < trainseqs.size(); i++) {
			Sequence seq = trainseqs.get(i);
			long user = seq.getUserId();
			if (!user2seqs.containsKey(user)) {
				user2seqs.put(user, new HashSet<Integer>());
			}
			user2seqs.get(user).add(i);
		}
		testSeqs = new ArrayList<Sequence>();
		for (int i = (int) (allSeqs.size() * (1 - testRatio)); i < allSeqs.size(); ++i) {
			Sequence seq = allSeqs.get(i);
			long user = seq.getUserId();
			if (user2seqs.containsKey(user)) {
				testSeqs.add(seq);
			}
		}
		System.out.println("filtered testSeqs size: " + testSeqs.size());
	}

	// add training seq
	public void addSequence(Sequence s) {
		this.trainseqs.add(s);
	}

	// add test seq
	public void addTestSequence(Sequence s) {
		this.testSeqs.add(s);
	}

	public void addTextDatum(Map<Integer, Integer> message) {
		this.textData.add(message);
	}

	public void addGeoDatum(RealVector rv) {
		this.geoData.add(rv);
	}

	public void addTemporalDatum(RealVector rv) {
		this.temporalData.add(rv);
	}


	public void setNumWords(int numWords) {
		this.numWords = numWords;
	}

	public void setTestRatio(double testRatio) {
		this.testRatio = testRatio;
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
		if (items.length < 6) {
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
		String[] items = s.split("\\s");
		if (items.length == 0) {
			System.out.println("Warning! Checkin has no message.");
		}
		for (int i = 0; i < items.length; i++) {
			int wordId = Integer.parseInt(items[i]);
			int oldCnt = message.containsKey(wordId) ? message.get(wordId) : 0;
			message.put(wordId, oldCnt + 1);
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

	public SequenceDataset getCopy() {
		SequenceDataset copiedDataSet = new SequenceDataset();
		for (Sequence s : trainseqs) {
			copiedDataSet.addSequence(s.copy());
		}
		for (Sequence s : testSeqs) {
			copiedDataSet.addTestSequence(s.copy());
		}
		for (Map<Integer, Integer> m : textData) {
			copiedDataSet.addTextDatum(new HashMap(m));
		}
		for (RealVector rv : geoData ) {
			copiedDataSet.addGeoDatum(rv);
		}
		for (RealVector rv : temporalData) {
			copiedDataSet.addTemporalDatum(rv);
		}
		copiedDataSet.setNumWords(this.numWords);
		copiedDataSet.setTestRatio(this.testRatio);
		return copiedDataSet;
	}

}
