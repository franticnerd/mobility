package predict;

import data.Checkin;
import data.PredictionDataset;
import data.Sequence;
import data.SequenceDataset;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import java.util.List;
import java.util.Set;

/**
 * An abstract class for predicting next location.
 * Created by chao on 5/3/15.
 */
public abstract class Predictor {

    double accuracy = 0;

    // Input: a database of length-2 movements;
    public void predict(PredictionDataset mdb, int K) {
        int numCorrectPrediction = 0;
        for (int i=0; i<mdb.size(); i++) {
            Sequence m = mdb.getSeq(i);
            Set<Checkin> candidate = mdb.getCands(i);
            TopKSearcher tks = new TopKSearcher();
            tks.init(K);
            for (Checkin p : candidate) {
                ScoreCell sc = calcScore(m, p);
                tks.add(sc);
            }
            ScoreCell [] topKResults = new ScoreCell[K];
            topKResults = tks.getTopKList(topKResults);
            if (isCorrect(m, topKResults)) {
                numCorrectPrediction ++;
//                System.out.println(candidate.size() + " +");
            } else {
//                System.out.println(candidate.size() + " -");
            }
        }
        accuracy = (double) numCorrectPrediction / (double) mdb.size();
    }

    public abstract ScoreCell calcScore(Sequence m, Checkin p);

    public boolean isCorrect(Sequence m, ScoreCell [] topKResult) {
        int groundTruth = m.getCheckin(1).getId();
        for (int i=0; i<topKResult.length; i++)
            if (groundTruth == topKResult[i].getId())
                return  true;
        return false;
    }

    public double getAccuracy() {
        return accuracy;
    }

}
