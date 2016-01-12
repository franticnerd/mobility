package demo;

import com.mongodb.*;
import model.*;
import predict.DistancePredictor;
import predict.HMMPredictor;
import predict.Predictor;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by chao on 7/7/15.
 */
public class Mongo {

    String host;
    int port;
    String db;
    String sequenceColName;
    String keywordColName;
    String modelColName;
    String expColName;

    public DBCollection sequenceCol;
    public DBCollection keywordCol;
    public DBCollection modelCol;
    public DBCollection expCol;

    public Mongo(Map config) throws Exception {
        host = (String) ((Map)config.get("mongo")).get("dns");
        port = (Integer) ((Map)config.get("mongo")).get("port");
        db = (String) ((Map)config.get("mongo")).get("db");
        sequenceColName = (String) ((Map)config.get("mongo")).get("sequences");
        keywordColName = (String) ((Map)config.get("mongo")).get("words");
        modelColName = (String) ((Map)config.get("mongo")).get("models");
        expColName = (String) ((Map)config.get("mongo")).get("exps");

        MongoClient mongoClient = new MongoClient(host, port);
        DB database = mongoClient.getDB(db);
        sequenceCol = database.getCollection(sequenceColName);
        keywordCol = database.getCollection(keywordColName);
        modelCol = database.getCollection(modelColName);
        expCol = database.getCollection(expColName);
    }

    public void writeModels(HMM h, GeoHMM g) {
        // get current time when finished running the experiments
        modelCol.remove(new BasicDBObject());
        modelCol.insert(new BasicDBObject("geohmm", g.toBson()));

//        modelCol.insert(new BasicDBObject("background", b.toBson()));
//        modelCol.insert(new BasicDBObject("mixture", m.toBson()));


    }

    public void writeHMM(HMM h, boolean augmentTest, double augmentThreshold) {
        modelCol.insert(new BasicDBObject("hmm", h.toBson())
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold));
    }

    public void writeGeoHMM(GeoHMM h, boolean augmentTest, double augmentThreshold) {
        modelCol.insert(new BasicDBObject("geohmm", h.toBson())
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold));
    }

    public HMM loadHMM(int numStates, boolean augmentTest, double augmentThreshold) {
        DBObject query = new BasicDBObject("hmm.K", numStates)
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold);
        DBObject doc = modelCol.findOne(query);
        return new HMM((DBObject)doc.get("hmm"));
    }

    public GeoHMM loadGeoHMM(int numStates, boolean augmentTest, double augmentThreshold) {
        DBObject query = new BasicDBObject("geohmm.K", numStates)
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold);
        DBObject doc = modelCol.findOne(query);
        return new GeoHMM((DBObject)doc.get("geohmm"));
    }


    public void writePrediction(DistancePredictor p, int K) {
        expCol.insert(new BasicDBObject("distance", null)
                .append("Accuracy", p.getAccuracy())
                .append("K", K));
    }

    public void writePredicton(HMM h, Predictor p, boolean augmentTest, double augmentThreshold, int K) {
        expCol.insert(new BasicDBObject("hmm", h.statsToBson())
                .append("Accuracy", p.getAccuracy())
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold)
                .append("K", K));
    }


    public void writePredicton(GeoHMM h, Predictor p, boolean augmentTest, double augmentThreshold, int K) {
        expCol.insert(new BasicDBObject("geohmm", h.statsToBson())
                .append("Accuracy", p.getAccuracy())
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold)
                .append("K", K));
    }

    public void writePredicton(EHMM h, Predictor p, boolean augmentTest, double augmentThreshold, int K) {
        modelCol.insert(new BasicDBObject("ehmm", h.statsToBson())
                .append("Accuracy", p.getAccuracy())
                .append("augment", augmentTest)
                .append("augmentThreshold", augmentThreshold)
                .append("K", K));
    }


    /** ---------------------------------- Main ---------------------------------- **/
    public static void main(String [] args) throws Exception {
        String paraFile = args.length > 0 ? args[0] : "../run/ny40k.yaml";
        Map config = new Config().load(paraFile);
        Mongo m = new Mongo(config);
//        m.modelCol.drop();
        System.out.println(m.modelCol.count());
        for (DBObject d : m.expCol.find())
            System.out.println(d);
//        HMM h = m.loadHMM(10, false, 0.1);
//        System.out.println(h);
    }


}

