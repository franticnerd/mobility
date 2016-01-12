package demo;

import java.util.*;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;

import data.Checkin;
import data.CheckinDataset;
import data.PredictionDataset;
import data.Sequence;
import data.SequenceDataset;
import data.WordDataset;
import model.Background;
import model.EHMM;
import model.HMM;
import model.Mixture;
import myutils.*;
import predict.DistancePredictor;
import predict.EHMMPredictor;
import predict.HMMPredictor;
import textAugmentation.*;

//entrance function used by Keyang for testing
public class Test {
	static public final String WorkPath = "/Users/keyangzhang/Documents/UIUC/Research/Mobility/mobility/";
	//	static public final String WorkPath = "/home/kzhang53/Mobility/mobility/";

	static Map config;
	static Mongo mongo;
	static Background b;
	static HMM h;
	static EHMM e;
	static Mixture m;

	static WordDataset wd = new WordDataset();
	static SequenceDataset hmmd = new SequenceDataset();
	static CheckinDataset bgd = new CheckinDataset();

	/**
	 * ---------------------------------- Initialize
	 * ----------------------------------
	 **/
	static void init(String paraFile) throws Exception {
		config = new Config().load(paraFile);
		mongo = new Mongo(config); // init the connection to mongo db.
		loadData();
	}

	static void loadData() throws Exception {
		// load data
		String wordFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("words");
		String sequenceFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("sequences");
		double testRatio = (Double) ((Map) config.get("predict")).get("testRatio");
		boolean filterTest = (Boolean) ((Map) config.get("predict")).get("filterTest");
		wd.load(wordFile);
		hmmd.load(sequenceFile, testRatio, filterTest);
		hmmd.setNumWords(wd.size());

		int LngGridNum = (Integer) ((Map) config.get("augment")).get("LngGridNum");
		int LatGridNum = (Integer) ((Map) config.get("augment")).get("LatGridNum");
		double SimilarityThresh = (Double) ((Map) config.get("augment")).get("SimilarityThresh");
		boolean augmentTrain = (Boolean) ((Map) config.get("augment")).get("augmentTrain");
		boolean augmentTest = (Boolean) ((Map) config.get("augment")).get("augmentTest");
		int augmentedSize = (Integer) ((Map) config.get("augment")).get("augmentedSize");
		Augmenter augmenter = new Augmenter(hmmd, wd, LngGridNum, LatGridNum, SimilarityThresh);
		hmmd.augmentText(augmenter, augmentedSize, augmentTrain, augmentTest);
		//        bgd.load(hmmd);
	}

	/**
	 * ---------------------------------- Train
	 * ----------------------------------
	 **/
	static void train() throws Exception {
		boolean isTrain = ((Boolean) ((Map) config.get("model")).get("train"));
		//      b = isTrain ? trainBackground() : mongo.loadBackground();
		h = trainHMM();
		e = trainEHMM();
		//      m = isTrain ? trainMixture() : mongo.loadMixture();
	}

	static Background trainBackground() {
		int maxIter = (Integer) ((Map) config.get("model")).get("maxIter");
		int numState = (Integer) ((Map) ((Map) config.get("model")).get("background")).get("numState");
		Background b = new Background(maxIter);
		b.train(bgd, numState);
		System.out.println("Finished training background model.");
		return b;
	}

	static HMM trainHMM() {
		int maxIter = (Integer) ((Map) config.get("model")).get("maxIter");
		int numState = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numState");
		int numComponent = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numComponent");
		HMM h = new HMM(maxIter);
		h.train(hmmd, 10, numComponent);
		System.out.println("Finished training HMM.");
		return h;
	}

	static EHMM trainEHMM() throws Exception {
		int MaxIter = (Integer) ((Map) config.get("model")).get("maxIter");
		int BG_numState = (Integer) ((Map) ((Map) config.get("model")).get("background")).get("numState");
		int HMM_K = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numState");
		int HMM_M = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numComponent");
		int C = (Integer) ((Map) ((Map) config.get("model")).get("ehmm")).get("numCluster");
		String initMethod = (String) ((Map) ((Map) config.get("model")).get("ehmm")).get("initMethod");
		EHMM e = new EHMM(MaxIter, BG_numState, HMM_K, HMM_M, C, initMethod);
		e.train(hmmd);
		System.out.println("Finished training MixtureOfHMMs.");
		return e;
	}

	static Mixture trainMixture() {
		int maxIter = (Integer) ((Map) config.get("model")).get("maxIter");
		int numState = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numState");
		int numComponent = (Integer) ((Map) ((Map) config.get("model")).get("hmm")).get("numComponent");
		Mixture m = new Mixture(maxIter, b);
		m.train(hmmd, numState, numComponent);
		System.out.println("Finished training the Mixture model.");
		return m;
	}


	/**
	 * ---------------------------------- Predict
	 * ----------------------------------
	 **/
	public static void predict() throws Exception {
		double distThre = (Double) ((Map) config.get("predict")).get("distThre");
		double timeThre = (Double) ((Map) config.get("predict")).get("timeThre");
		int K = (Integer) ((Map) config.get("predict")).get("K");
		boolean avgTest = (Boolean) ((Map) config.get("predict")).get("avgTest");
		PredictionDataset pd = hmmd.extractTestData();
		pd.genCandidates(distThre, timeThre);
		DistancePredictor dp = new DistancePredictor();
		dp.predict(pd, K);
		dp.printAccuracy();
		HMMPredictor hp = new HMMPredictor(h, avgTest);
		hp.predict(pd, K);
		hp.printAccuracy();
		EHMMPredictor ep = new EHMMPredictor(e, avgTest);
		ep.predict(pd, K);
		ep.printAccuracy();
		//        HMMPredictor mp = new HMMPredictor(m);
		//        mp.predict(pd, K);
		//        mp.printAccuracy();
	}

	/**
	 * ---------------------------------- Main
	 * ----------------------------------
	 **/
	public static void main(String[] args) throws Exception {
		System.setOut(new PrintStream(new FileOutputStream(Test.WorkPath + "results/result.txt")));
		//        String paraFile = args.length > 0 ? args[0] : "../run/4sq.yaml";
		//		    	String paraFile = args.length > 0 ? args[0] : Test.WorkPath+"run/4sq.yaml";
		String paraFile = args.length > 0 ? args[0] : Test.WorkPath + "run/ny40k.yaml";
		init(paraFile);
		train();
		//		writeModels();
		predict();
	}

	public static void testAugmenter(String[] args) throws Exception {
		System.setOut(new PrintStream(new FileOutputStream(WorkPath + "results/result.txt")));
		String datasetName = "ny40k.yaml";
		//		String datasetName = "4sq.yaml";
		String paraFile = args.length > 0 ? args[0] : WorkPath + "run/" + datasetName;

		Map config = new Config().load(paraFile);
		WordDataset wd = new WordDataset();
		SequenceDataset hmmd = new SequenceDataset();
		String wordFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("words");
		String sequenceFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("sequences");
		wd.load(wordFile);
		hmmd.load(sequenceFile);
		hmmd.setNumWords(wd.size());

		WordSimilarity ws = new WordSimilarity(hmmd, wd, 10, 10, 0.1);
		//		ws.printHighlySimilarPairs();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(WorkPath + "results/similarity.dat"));
		oos.writeObject(ws);
		oos.flush();

		Augmenter augmentation = new Augmenter(hmmd, wd, WorkPath + "results/similarity.dat");
		for (Sequence sequence : hmmd.getSequences()) {
			for (Checkin checkin : sequence.getCheckins()) {
				Map<Integer, Integer> text = checkin.getMessage();
				Map<Integer, Integer> augmentedText = augmentation.getAugmentedText(text, 30);
				//				checkin.setMessage(augmentedText);
				for (int word : text.keySet()) {
					System.out.print(wd.getWord(word) + "=" + text.get(word) + ", ");
				}
				System.out.println();
				for (int word : augmentedText.keySet()) {
					System.out.print(wd.getWord(word) + "=" + augmentedText.get(word) + ", ");
				}
				System.out.println("\n");
			}
		}
	}
}
