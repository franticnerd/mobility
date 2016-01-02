package demo;

import data.CheckinDataset;
import data.PredictionDataset;
import data.SequenceDataset;
import data.WordDataset;
import model.Background;
import model.HMM;
import model.Mixture;
import predict.DistancePredictor;
import predict.HMMPredictor;

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
    static Mixture m;

    static WordDataset wd = new WordDataset();
    static SequenceDataset hmmd = new SequenceDataset();
    static CheckinDataset bgd = new CheckinDataset();

    /** ---------------------------------- Initialize ---------------------------------- **/
    static void init(String paraFile) throws Exception {
        config = new Config().load(paraFile);
        mongo = new Mongo(config); // init the connection to mongo db.
        loadData();
    }

    static void loadData() throws Exception {
        // load data
        String wordFile = (String) ((Map)((Map)config.get("file")).get("input")).get("words");
        String sequenceFile = (String) ((Map)((Map)config.get("file")).get("input")).get("sequences");
        double testRatio = (Double)((Map)config.get("predict")).get("testRatio");
        wd.load(wordFile);
        hmmd.load(sequenceFile, testRatio);
        hmmd.setNumWords(wd.size());
        bgd.load(hmmd);
    }

    /** ---------------------------------- Train ---------------------------------- **/
    static void train() throws Exception {
      boolean isTrain = ((Boolean)((Map)config.get("model")).get("train"));
      b = isTrain ? trainBackground() : mongo.loadBackground();
      h = isTrain ? trainHMM() : mongo.loadHMM();
      m = isTrain ? trainMixture() : mongo.loadMixture();
    }

    static Background trainBackground() {
      int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
      int numState = (Integer)((Map)((Map)config.get("model")).get("background")).get("numState");
      Background b = new Background(maxIter);
      b.train(bgd, numState);
      System.out.println("Finished training background model.");
      return b;
    }

    static HMM trainHMM() {
        int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
        int numState = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numState");
        int numComponent = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numComponent");
        HMM h = new HMM(maxIter);
        h.train(hmmd, numState, numComponent);
        System.out.println("Finished training HMM.");
        return h;
    }

    static Mixture trainMixture() {
        int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
        int numState = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numState");
        int numComponent = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numComponent");
        Mixture m = new Mixture(maxIter, b);
        m.train(hmmd, numState, numComponent);
        System.out.println("Finished training the Mixture model.");
        return m;
    }

    static void writeModels() throws Exception {
        if((Boolean)((Map)config.get("file")).get("write")) {
            b.write(wd, (String) ((Map) ((Map) config.get("post")).get("keyword")).get("bgd_description"));
            h.write(wd, (String)((Map)((Map)config.get("post")).get("keyword")).get("hmm_description"));
            m.write(wd, (String) ((Map) ((Map) config.get("post")).get("keyword")).get("mix_description"));
        }
        if((Boolean)((Map)config.get("mongo")).get("write")) {
            mongo.writeModels(b, h, m);
        }
    }

    /** ---------------------------------- Predict ---------------------------------- **/
    public static void predict() throws Exception {
        double distThre = (Double)((Map)config.get("predict")).get("distThre");
        double timeThre = (Double)((Map)config.get("predict")).get("timeThre");
        int K = (Integer)((Map)config.get("predict")).get("K");
        PredictionDataset pd = hmmd.extractTestData();
        pd.genCandidates(distThre, timeThre);
        DistancePredictor dp = new DistancePredictor();
        dp.predict(pd, K);
        dp.printAccuracy();
        HMMPredictor hp = new HMMPredictor(h);
        hp.predict(pd, K);
        hp.printAccuracy();
        HMMPredictor mp = new HMMPredictor(m);
        mp.predict(pd, K);
        mp.printAccuracy();
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

