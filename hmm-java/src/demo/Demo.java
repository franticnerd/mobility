package demo;

import data.CheckinDataset;
import data.PredictionDataset;
import data.SequenceDataset;
import data.WordDataset;
import model.Background;
import model.EHMM;
import model.HMM;
import model.Mixture;
import predict.DistancePredictor;
import predict.EHMMPredictor;
import predict.HMMPredictor;
import textAugmentation.Augmenter;
import textAugmentation.WordSimilarity;

import java.util.Map;

/**
 * The main file for evaluating the models.
 * Created by chao on 4/16/15.
 */
public class Demo {

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
		h = isTrain ? trainHMM() : mongo.loadHMM();
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

	static void writeModels() throws Exception {
		if ((Boolean) ((Map) config.get("file")).get("write")) {
			//            b.write(wd, (String) ((Map) ((Map) config.get("post")).get("keyword")).get("bgd_description"));
			h.write(wd, (String) ((Map) ((Map) config.get("post")).get("keyword")).get("hmm_description"));
			//            m.write(wd, (String) ((Map) ((Map) config.get("post")).get("keyword")).get("mix_description"));
		}
		if ((Boolean) ((Map) config.get("mongo")).get("write")) {
			mongo.writeModels(b, h, m);
		}
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

    /** ---------------------------------- Main ---------------------------------- **/
    public static void main(String [] args) throws Exception {
        String paraFile = args.length > 0 ? args[0] : "../run/ny40k.yaml";
        init(paraFile);
        train();
        writeModels();
        predict();
    }

}
