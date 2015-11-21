package demo;

import com.mongodb.*;
import model.Background;
import model.HMM;
import model.Mixture;

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

    public DBCollection sequenceCol;
    public DBCollection keywordCol;
    public DBCollection modelCol;

    public Mongo(Map config) throws Exception {
        host = (String) ((Map)config.get("mongo")).get("dns");
        port = (Integer) ((Map)config.get("mongo")).get("port");
        db = (String) ((Map)config.get("mongo")).get("db");
        sequenceColName = (String) ((Map)config.get("mongo")).get("sequences");
        keywordColName = (String) ((Map)config.get("mongo")).get("words");
        modelColName = (String) ((Map)config.get("mongo")).get("models");

        MongoClient mongoClient = new MongoClient(host, port);
        DB database = mongoClient.getDB(db);
        sequenceCol = database.getCollection(sequenceColName);
        keywordCol = database.getCollection(keywordColName);
        modelCol = database.getCollection(modelColName);
    }

    public void writeModels(Background b, HMM h, Mixture m) {
        // get current time when finished running the experiments
        modelCol.remove(new BasicDBObject());
        modelCol.insert(new BasicDBObject("background", b.toBson()));
        modelCol.insert(new BasicDBObject("hmm", h.toBson()));
        modelCol.insert(new BasicDBObject("mixture", m.toBson()));

//        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
//        Date date = new Date();
//        BasicDBObject doc = new BasicDBObject().append("time", dateFormat.format(date));
//        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
//        Date date = new Date();
//        BasicDBObject doc = new BasicDBObject().append("time", dateFormat.format(date));
////        doc.append("background", b.toBson());
////        doc.append("hmm", h.toBson());
//        doc.append("mixture", m.toBson());
//        modelCol.insert(doc);

    }

    public Background loadBackground() {
        DBObject query = new BasicDBObject("background", new BasicDBObject("$exists", true));
        DBObject doc = modelCol.findOne(query);
        return new Background((DBObject)doc.get("background"));
    }

    public HMM loadHMM() {
        DBObject query = new BasicDBObject("hmm", new BasicDBObject("$exists", true));
        DBObject doc = modelCol.findOne(query);
        return new HMM((DBObject)doc.get("hmm"));
    }

    public Mixture loadMixture() {
        DBObject bgquery = new BasicDBObject("background", new BasicDBObject("$exists", true));
        DBObject background = modelCol.findOne(bgquery);
        DBObject query = new BasicDBObject("mixture", new BasicDBObject("$exists", true));
        DBObject doc = modelCol.findOne(query);
        return new Mixture((DBObject)background.get("background"), (DBObject)doc.get("mixture"));
    }

}

