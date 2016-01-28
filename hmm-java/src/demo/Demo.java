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

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * The main file for evaluating the models.
 * Created by chao on 4/16/15.
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
	static List<Integer> numStateList;  // the number of states for HMM.
	static int numComponent; // the number of GMM component for HMM
	static List<Integer> numClusterList; // the number of clusters for EHMM.
	static List<String> initMethodList;  // the list of initalization methods for EHMM.

	// parameters for augmentation.
	static List<Double> thresholdList;  // the list of similarity thresholds for augmenting text
	static List<Integer> augmentSizeList;  // the list of augmentation size
	static List<Integer> numAxisBinList;  // the list of number of bins per axis
	static boolean augmentTrain;
	static boolean augmentTest;
	static double augmentThreshold;
	static int augmentSize;
	static int numAxisBin;

	// parameters for prediction
	static double distThre;
	static double timeThre;
	static boolean filterTest;

	/**
	 * ---------------------------------- Initialize ----------------------------------
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
		filterTest = (Boolean) ((Map) config.get("predict")).get("filterTest");
		wd.load(wordFile);
		rawDb.load(sequenceFile, testRatio, filterTest);
		rawDb.setNumWords(wd.size());

		// augment the text data by mining word spatiotemporal correlations.
		thresholdList = (List<Double>) ((Map) config.get("augment")).get("threshold");
		augmentSizeList = (List<Integer>) ((Map) config.get("augment")).get("augmentedSize");
		numAxisBinList = (List<Integer>) ((Map) config.get("augment")).get("numAxisBin");
		augmentTrain = (Boolean) ((Map) config.get("augment")).get("augmentTrain");
		augmentTest = (Boolean) ((Map) config.get("augment")).get("augmentTest");
		augmentSize = augmentSizeList.get(0);
		augmentThreshold = thresholdList.get(0);
		numAxisBin = numAxisBinList.get(0);
        distThre = (Double) ((Map) config.get("predict")).get("distThre");
        timeThre = (Double) ((Map) config.get("predict")).get("timeThre");
		// generate the hmmd and prediction data
		getAugmentedDataSet();

		// the model parameters
		maxIter = (Integer) ((Map) config.get("hmm")).get("maxIter");
		KList = (List<Integer>) ((Map) config.get("predict")).get("K");
		avgTest = (Boolean) ((Map) config.get("predict")).get("avgTest");
		numStateList = (List<Integer>) ((Map) config.get("hmm")).get("numState");
		numComponent = (Integer) ((Map) config.get("hmm")).get("numComponent");
		numClusterList = (List<Integer>) ((Map) config.get("ehmm")).get("numCluster");
		initMethodList = (List<String>) ((Map) config.get("ehmm")).get("initMethod");
	}


	static void getAugmentedDataSet() throws Exception {
		Augmenter augmenter = new Augmenter(rawDb, wd, numAxisBin, numAxisBin, augmentThreshold);
		// augmented data for training
		hmmdb = rawDb.getCopy();
		hmmdb.augmentText(augmenter, augmentSize, augmentTrain, augmentTest);
		// augmented data for testing
		pd = hmmdb.extractTestData();
		pd.genCandidates(distThre, timeThre);
	}


	/**
	 * ---------------------------------- Train and Predict ----------------------------------
	 **/
	static void run() throws Exception {
		// run the predictors using default parameters
		runDistance();
		runGeoHMM(maxIter, numStateList.get(0), numComponent);
		runHMM(maxIter, numStateList.get(0), numComponent);
        runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
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
			System.out.println("Distance based prediction accuracy:" + dp.getAccuracy());
			mongo.writePrediction(dp, K);
		}
	}



	static void runGeoHMM(int maxIter, int numStates, int numComponent) throws Exception {
		GeoHMM geoHMM;
		try {
			geoHMM =  mongo.loadGeoHMM(numStates);
//			throw new IOException();
		} catch (Exception e) {
			System.out.println("Cannot load GeoHMM from the Mongo DB. Start GeoHMM training.");
			geoHMM = new GeoHMM(maxIter);
			geoHMM.train(hmmdb, numStates, numComponent);
			mongo.writeGeoHMM(geoHMM);
		}
		// predict
		HMMPredictor hp = new HMMPredictor(geoHMM, avgTest);
		for (Integer K : KList) {
			hp.predict(pd, K);
			System.out.println("GeoHMM based prediction accuracy:" + hp.getAccuracy());
			mongo.writePredicton(geoHMM, hp, K);
		}
	}


	static void runHMM(int maxIter, int numStates, int numComponent) throws Exception {
		HMM h;
		try {
			h = mongo.loadHMM(numStates, augmentTest, augmentThreshold, augmentSize, numAxisBin);
//			throw new IOException();
		} catch (Exception e) {
			System.out.println("Cannot load HMM from the Mongo DB. Start HMM training.");
			h = new HMM(maxIter);
			h.train(hmmdb, numStates, numComponent);
			mongo.writeHMM(h, augmentTest, augmentThreshold, augmentSize, numAxisBin);
		}
		// predict
		HMMPredictor hp = new HMMPredictor(h, avgTest);
		for (Integer K : KList) {
			hp.predict(pd, K);
			System.out.println("HMM based prediction accuracy:" + hp.getAccuracy());
			mongo.writePredicton(h, hp, augmentTest, augmentThreshold, augmentSize, numAxisBin, K);
		}
	}

	static void runEHMM(int maxIter, int numCluster, int numStates, int numComponent, String initMethod) throws Exception {
		EHMM ehmm;
		try {
            ehmm =  mongo.loadEHMM(numStates, numCluster, initMethod, hmmdb,
                        augmentTest, augmentThreshold, augmentSize, numAxisBin);
		} catch (Exception e) {
			System.out.println("Cannot load EHMM from the Mongo DB. Start EHMM training.");
			ehmm = new EHMM(maxIter, numStates, numStates, numComponent, numCluster, initMethod);
			ehmm.train(hmmdb);
			mongo.writeEHMM(ehmm, augmentTest, augmentThreshold, augmentSize, numAxisBin);
		}
		EHMMPredictor ep = new EHMMPredictor(ehmm, avgTest);
		for (Integer K : KList) {
			ep.predict(pd, K);
			System.out.println("EHMM based prediction accuracy:" + ep.getAccuracy());
			mongo.writePredicton(ehmm, ep, augmentTest, augmentThreshold, augmentSize, numAxisBin, K);
		}
	}

	/**
	 * Evaluate different parameters.
	 */
	static void evalNumStates() throws Exception {
		if ((Boolean) ((Map)config.get("hmm")).get("evalNumState") == false)	return;
		for (Integer numState : numStateList) {
			runHMM(maxIter, numState, numComponent);
			runGeoHMM(maxIter, numState, numComponent);
			runEHMM(maxIter, numClusterList.get(0), numState, numComponent, initMethodList.get(0));
		}
	}

	static void evalNumCluster() throws Exception {
		if ((Boolean) ((Map)config.get("ehmm")).get("evalNumCluster") == false)	return;
		for (Integer numCluster : numClusterList) {
			runEHMM(maxIter, numCluster, numStateList.get(0), numComponent, initMethodList.get(0));
		}
	}


	static void evalInitMethod() throws Exception {
		if ((Boolean) ((Map)config.get("ehmm")).get("evalInitMethod") == false)	return;
		for (String initMethod : initMethodList) {
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethod);
		}
	}

	static void evalAugmentation() throws Exception {
		evalAugmentationThresh();
		evalAugmentationSize();
		evalNumAxisBin();
	}


	static void evalAugmentationThresh() throws Exception {
		if ((Boolean) ((Map)config.get("augment")).get("evalThresh") == false)	return;
		for (Double threshold : thresholdList) {
			augmentThreshold = threshold;
			getAugmentedDataSet();
			runHMM(maxIter, numStateList.get(0), numComponent);
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
		}
		augmentThreshold = thresholdList.get(0);  // restore the default value
	}

	static void evalAugmentationSize() throws Exception {
		if ((Boolean) ((Map)config.get("augment")).get("evalSize") == false)	return;
		for (Integer asize : augmentSizeList) {
			augmentSize = asize;
			getAugmentedDataSet();
			runHMM(maxIter, numStateList.get(0), numComponent);
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
		}
		augmentSize = augmentSizeList.get(0);
	}

	static void evalNumAxisBin() throws Exception {
		if ((Boolean) ((Map)config.get("augment")).get("evalNumBin") == false)	return;
		for (Integer numBin : numAxisBinList) {
			numAxisBin = numBin;
			getAugmentedDataSet();
			runHMM(maxIter, numStateList.get(0), numComponent);
			runEHMM(maxIter, numClusterList.get(0), numStateList.get(0), numComponent, initMethodList.get(0));
		}
		numAxisBin = numAxisBinList.get(0);
	}

    /** ---------------------------------- Main ---------------------------------- **/
    public static void main(String [] args) throws Exception {
//        String paraFile = args.length > 0 ? args[0] : "../run/ny40k.yaml";
		String paraFile = args.length > 0 ? args[0] : "../run/tweet-limited.yaml";
        init(paraFile);
		run();
    }

}

