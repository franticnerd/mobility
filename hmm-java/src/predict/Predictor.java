package predict;

import data.Movement;
import data.MovementDB;
import data.Place;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import java.util.List;

/**
 * An abstract class for predicting next location.
 * Created by chao on 5/3/15.
 */
public abstract class Predictor {

    double accuracy = 0;


    // Input: a database of length-2 movements;
    public void predict(MovementDB mdb, int K) {
        int numCorrectPrediction = 0;
        for (int i=0; i<mdb.size(); i++) {
            Movement m = mdb.getMovement(i);
            List<Place> candidate = mdb.getCandidate(i);
            TopKSearcher tks = new TopKSearcher();
            tks.init(K);
            for (Place p : candidate) {
                ScoreCell sc = calcScore(m, p);
                tks.add(sc);
            }
            ScoreCell [] topKResults = new ScoreCell[K];
            topKResults = tks.getTopKList(topKResults);
            if (isCorrect(m, topKResults)) {
                numCorrectPrediction ++;
                System.out.println(candidate.size() + " +");
            } else {
                System.out.println(candidate.size() + " -");
            }
        }
        accuracy = (double) numCorrectPrediction / (double) mdb.size();
    }

    public abstract ScoreCell calcScore(Movement m, Place p);

    public boolean isCorrect(Movement m, ScoreCell [] topKResult) {
        int groundTruth = m.getEndPlace().getId();
        for (int i=0; i<topKResult.length; i++)
            if (groundTruth == topKResult[i].getId())
                return  true;
        return false;
    }

    public double getAccuracy() {
        return accuracy;
    }

}
