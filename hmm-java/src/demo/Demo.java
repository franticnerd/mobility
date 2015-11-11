package demo;

import data.BackgroundDatabase;
import data.Database;
import data.HMMDatabase;
import model.Background;
import model.HMM;
import model.Mixture;

import java.util.Map;

/**
 * The main file for evaluating the models.
 * Created by chao on 4/16/15.
 */
public class Demo {

    static Map config;
    static Background b;
    static HMM h;
    static Mixture m;
    static Mongo mongo;

    /** ---------------------------------- Initialize ---------------------------------- **/
    static void init(String paraFile) throws Exception {
        config = new Config().load(paraFile);
        mongo = new Mongo(config); // init the connection to mongo db.
        loadData();
    }

    static void loadData() throws Exception {
        // load data
        String placeFile = (String) ((Map)((Map)config.get("file")).get("input")).get("places");
        String checkinFile = (String) ((Map)((Map)config.get("file")).get("input")).get("checkins");
        String wordFile = (String) ((Map)((Map)config.get("file")).get("input")).get("words");
        Database.loadData(placeFile, checkinFile, wordFile);
    }

    /** ---------------------------------- Train ---------------------------------- **/
    static void train() throws Exception {
      boolean isTrain = ((Boolean)((Map)config.get("model")).get("train"));
      b = isTrain ? trainBackground() : mongo.loadBackground();
      h = isTrain ? trainHMM() : mongo.loadHMM();
      m = isTrain ? trainMixture() : mongo.loadMixture();
    }

    static Background trainBackground() {
      BackgroundDatabase bgd = new BackgroundDatabase();
      bgd.load();  // extract the background training data from the database.
      int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
      int numState = (Integer)((Map)((Map)config.get("model")).get("background")).get("numState");
      Background b = new Background(maxIter);
      b.train(bgd, numState);
      System.out.println("Finished training background model.");
      return b;
    }

    static HMM trainHMM() {
        HMMDatabase hmmd = new HMMDatabase();
        int minGap = (Integer)((Map)config.get("preprocess")).get("minGap");
        int maxGap = (Integer)((Map)config.get("preprocess")).get("maxGap");
        hmmd.load(minGap, maxGap);
        int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
        int numState = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numState");
        int numComponent = (Integer)((Map)((Map)config.get("model")).get("hmm")).get("numComponent");
        HMM h = new HMM(maxIter);
        h.train(hmmd, numState, numComponent);
        System.out.println("Finished training HMM.");
        return h;
    }

    static Mixture trainMixture() {
        HMMDatabase hmmd = new HMMDatabase();
        int minGap = (Integer)((Map)config.get("preprocess")).get("minGap");
        int maxGap = (Integer)((Map)config.get("preprocess")).get("maxGap");
        hmmd.load(minGap, maxGap);
        int maxIter = (Integer)((Map)config.get("model")).get("maxIter");
        int numState = (Integer)((Map)((Map)config.get("model")).get("mixture")).get("numState");
        int numComponent = (Integer)((Map)((Map)config.get("model")).get("mixture")).get("numComponent");
        Mixture m = new Mixture(maxIter, b);
        m.train(hmmd, numState, numComponent);
        System.out.println("Finished training the Mixture model.");
        return m;
    }

    static void writeModels() throws Exception {
        if((Boolean)((Map)config.get("file")).get("write")) {
            b.write((String) ((Map) ((Map) config.get("file")).get("output")).get("bg_description"));
            h.write((String)((Map)((Map)config.get("file")).get("output")).get("hmm_description"));
            m.write((String)((Map)((Map)config.get("file")).get("output")).get("mixture_description"));
        }
        if((Boolean)((Map)config.get("mongo")).get("write")) {
            mongo.writeModels(b, h, m);
        }
    }


    /** ---------------------------------- Main ---------------------------------- **/
    public static void main(String [] args) throws Exception {
        String paraFile = args.length > 0 ? args[0] : "../run/4sq.yaml";
        init(paraFile);
        train();
        writeModels();
    }

}


// /** ---------------------------------- Predict ---------------------------------- **/
//     public static void predict(HMM h, Mixture m) throws Exception {
//         genTestSequences(pc);
//         int K = pc.getIntPara("numPrediction");
//         MovementDB mdb = new MovementDB();
//         mdb.load(pc.getStringPara("labeledSeqFile"));
//         mdb.genCandidates();
//         PopularityPredictor pp = new PopularityPredictor();
//         pp.predict(mdb, K);
//         pp.printAccuracy();
//         DistancePredictor dp = new DistancePredictor();
//         dp.predict(mdb, K);
//         dp.printAccuracy();
//         HMMPredictor hp = new HMMPredictor(h);
//         hp.predict(mdb, K);
//         hp.printAccuracy();
//         HMMPredictor mp = new HMMPredictor(m);
//         mp.predict(mdb, K);
//         mp.printAccuracy();
//     }

//     public static void genTestSequences(ParaConfig pc) throws Exception {
//         HMMDatabase hmmd = new HMMDatabase();
//         hmmd.load(pc);
//         hmmd.writeTestSequences(pc.getStringPara("labeledSeqFile"), pc.getIntPara("numLabelSequence"));
//     }

//     /** ---------------------------------- Outlier ---------------------------------- **/
//     public static void detectOutlier(ParaConfig pc, HMM h, Mixture m) throws Exception {
//         int K = pc.getIntPara("numOutlier");
//         MovementDB mdb = new MovementDB();
//         mdb.load(pc.getStringPara("labeledSeqFile"));
//         mdb.genCandidates();
//         HMMDetector hd = new HMMDetector(m);
//         hd.detect(mdb, K);
//         hd.printOutliers();
//     }

//    static Background loadBackground() throws Exception {
//        if ((((Map)config.get("model")).get("loadSource")).equals("db")) {
//            return mongo.loadBackground();
//        } else {
//            String modelFile = (String) ((Map) ((Map) config.get("file")).get("output")).get("bg_model");
//            Background b = Background.load(modelFile);
//            System.out.println("Finished training background model.");
//            return b;
//        }
//    }
//
//    static HMM loadHMM() throws Exception {
//        if ((((Map)config.get("model")).get("loadSource")).equals("db")) {
//            return mongo.loadHMM();
//        } else {
//            String modelFile = (String) ((Map) ((Map) config.get("file")).get("output")).get("hmm_model");
//            HMM h = HMM.load(modelFile);
//            System.out.println("Finished loading HMM.");
//            return h;
//        }
//    }
//    static Mixture loadMixture() throws Exception {
//        if ((((Map)config.get("model")).get("loadSource")).equals("db")) {
//            return mongo.loadMixture();
//        } else {
//            String modelFile = (String) ((Map) ((Map) config.get("file")).get("output")).get("mixture_model");
//            Mixture m = Mixture.load(modelFile);
//            System.out.println("Finished loading the Mixture model.");
//            return m;
//        }
//    }
