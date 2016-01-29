package demo;

import data.PredictionDataset;
import data.SequenceDataset;
import data.WordDataset;
import model.EHMM;
import model.GeoHMM;
import model.HMM;
import predict.DistancePredictor;
import predict.EHMMPredictor;
import predict.HMMPredictor;
import textAugmentation.Augmenter;

import java.util.List;
import java.util.Map;

/**
 * The main file for evaluating the models. Created by chao on 4/16/15.
 */
public class Demo {

	static Map config;
	static Mongo mongo;

	static WordDataset wd = new WordDataset();
	static SequenceDataset rawDb = new SequenceDataset(); // the database before augmentation
	static SequenceDataset hmmdb = new SequenceDataset(); // the database after augmentation
	static PredictionDataset pd;

	static List<Integer> KList;
	static int maxIter;
	static boolean avgTest;
	static List<Integer> numStateList; // the number of states for HMM.
	static int numComponent; // the number of GMM component for HMM
	static List<Integer> numClusterList; // the number of clusters for EHMM.
	static List<String> initMethodList; // the list of initalization methods for EHMM.

	// parameters for augmentation.
	static int numAxisBin;
	static boolean augmentTrain;
	static boolean augmentTest;
	static int augmentedSize;
	static List<Double> thresholdList; // the list of similarity thresholds for augmenting text
	static double augmentThreshold;

	// parameters for prediction
	static double distThre;
	static double timeThre;

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
		rawDb.load(sequenceFile, testRatio, filterTest);
		rawDb.setNumWords(wd.size());

		// augment the text data by mining word spatiotemporal correlations.
		thresholdList = (List<Double>) ((Map) config.get("augment")).get("threshold");
		numAxisBin = (Integer) ((Map) config.get("augment")).get("numAxisBin");
		augmentTrain = (Boolean) ((Map) config.get("augment")).get("augmentTrain");
		augmentTest = (Boolean) ((Map) config.get("augment")).get("augmentTest");
		augmentedSize = (Integer) ((Map) config.get("augment")).get("augmentedSize");
		augmentThreshold = thresholdList.get(0);
		Augmenter augmenter = new Augmenter(rawDb, wd, numAxisBin, numAxisBin, augmentThreshold);
		hmmdb = rawDb.getCopy();
		hmmdb.augmentText(augmenter, augmentedSize, augmentTrain, augmentTest);

		// generate test sequences for location prediction.
		distThre = (Double) ((Map) config.get("predict")).get("distThre");
		timeThre = (Double) ((Map) config.get("predict")).get("timeThre");
		pd = hmmdb.extractTestData();
		pd.genCandidates(distThre, timeThre);

		// the model parameters
		maxIter = (Integer) ((Map) config.get("hmm")).get("maxIter");
		KList = (List<Integer>) ((Map) config.get("predict")).get("K");
		avgTest = (Boolean) ((Map) config.get("predict")).get("avgTest");
		numStateList = (List<Integer>) ((Map) config.get("hmm")).get("numState");
		numComponent = (Integer) ((Map) config.get("hmm")).get("numComponent");
		numClusterList = (List<Integer>) ((Map) config.get("ehmm")).get("numCluster");
		initMethodList = (List<String>) ((Map) config.get("ehmm")).get("initMethod");
	}

	/**
	 * ---------------------------------- Train and Predict
	 * ----------------------------------
	 **/
	static void run() throws Exception {
		// run the predictors using default parameters
//		runDistance();
		//		runHMM(maxIter, numStateList.get(0), numComponent);
		//		runGeoHMM(maxIter, numStateList.get(0), numComponent);
		//        runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
		// tune the parameters
		evalNumStates();
		evalNumCluster();
		evalInitMethod();
		evalAugmentation();
	}

	/**
	 * Run models with default paramters
	 */
	static void runDistance() {
		DistancePredictor dp = new DistancePredictor();
		for (Integer K : KList) {
			dp.predict(pd, K);
			dp.printAccuracy();
			mongo.writePrediction(dp, K);
		}
	}

	static void runHMM(int maxIter, int numStates, int numComponent) throws Exception {
		HMM h;
		try {
			h = mongo.loadHMM(numStates, augmentTest, augmentThreshold);
		} catch (Exception e) {
			System.out.println("Cannot load HMM from the Mongo DB. Start HMM training.");
			h = new HMM(maxIter);
			h.train(hmmdb, numStates, numComponent);
			mongo.writeHMM(h, augmentTest, augmentThreshold);
			// predict
			HMMPredictor hp = new HMMPredictor(h, avgTest);
			for (Integer K : KList) {
				hp.predict(pd, K);
				hp.printAccuracy();
				mongo.writePredicton(h, hp, augmentTest, augmentThreshold, K);
			}
		}
	}

	static void runGeoHMM(int maxIter, int numStates, int numComponent) throws Exception {
		GeoHMM geoHMM;
		try {
			geoHMM = mongo.loadGeoHMM(numStates, augmentTest, augmentThreshold);
		} catch (Exception e) {
			System.out.println("Cannot load GeoHMM from the Mongo DB. Start GeoHMM training.");
			geoHMM = new GeoHMM(maxIter);
			geoHMM.train(hmmdb, numStates, numComponent);
			mongo.writeGeoHMM(geoHMM, augmentTest, augmentThreshold);
			// predict
			HMMPredictor hp = new HMMPredictor(geoHMM, avgTest);
			for (Integer K : KList) {
				hp.predict(pd, K);
				hp.printAccuracy();
				mongo.writePredicton(geoHMM, hp, augmentTest, augmentThreshold, K);
			}
		}
	}

	static void runEHMM(int maxIter, int numCluster, int numStates, int numComponent, String initMethod)
			throws Exception {
		EHMM ehmm;
		try {
			ehmm = mongo.loadEHMM(numStates, numCluster, initMethod, augmentTest, augmentThreshold, hmmdb);
		} catch (Exception e) {
			System.out.println("Cannot load EHMM from the Mongo DB. Start EHMM training.");
			ehmm = new EHMM(maxIter, numStates, numStates, numComponent, numCluster, initMethod);
			ehmm.train(hmmdb);
			mongo.writeEHMM(ehmm, augmentTest, augmentThreshold);
			EHMMPredictor ep = new EHMMPredictor(ehmm, avgTest);
			for (Integer K : KList) {
				ep.predict(pd, K);
				ep.printAccuracy();
				mongo.writePredicton(ehmm, ep, augmentTest, augmentThreshold, K);
			}
		}
	}

	/**
	 * Evaluate different parameters.
	 */
	static void evalNumStates() throws Exception {
		if ((Boolean) ((Map) config.get("hmm")).get("evalNumState") == false)
			return;
		for (Integer numState : numStateList) {
			runHMM(maxIter, numState, numComponent);
			runGeoHMM(maxIter, numState, numComponent);
			runEHMM(maxIter, numClusterList.get(0), numState, numComponent, initMethodList.get(0));
		}
	}

	static void evalNumCluster() throws Exception {
		if ((Boolean) ((Map) config.get("ehmm")).get("evalNumCluster") == false)
			return;
		for (Integer numCluster : numClusterList) {
			runEHMM(maxIter, numCluster, numStateList.get(0), numComponent, initMethodList.get(0));
		}
	}

	static void evalInitMethod() throws Exception {
		if ((Boolean) ((Map) config.get("ehmm")).get("evalInitMethod") == false)
			return;
		for (String initMethod : initMethodList) {
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethod);
		}
	}

	/**
	 * ToDo: need to return a deep copy of the hmmdb for various settings.
	 */
	static void evalAugmentation() throws Exception {
		if ((Boolean) ((Map) config.get("augment")).get("evalThresh") == false)
			return;
		for (Double threshold : thresholdList) {
			augmentThreshold = threshold;
			Augmenter augmenter = new Augmenter(rawDb, wd, numAxisBin, numAxisBin, augmentThreshold);
			hmmdb = rawDb.getCopy();
			hmmdb.augmentText(augmenter, augmentedSize, augmentTrain, augmentTest);
			pd = hmmdb.extractTestData();
			pd.genCandidates(distThre, timeThre);
			runHMM(maxIter, numStateList.get(0), numComponent);
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
		}
	}

	/**
	 * ---------------------------------- Main
	 * ----------------------------------
	 **/
	public static void main(String[] args) throws Exception {
		String paraFile = args.length > 0 ? args[0] : "../run/ny40k.yaml";
		init(paraFile);
		run();
	}

}
