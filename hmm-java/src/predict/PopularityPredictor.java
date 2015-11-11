package predict;

import data.Movement;
import data.Place;
import myutils.ScoreCell;

/**
 * Created by chao on 5/3/15.
 */
public class PopularityPredictor extends Predictor {

    public ScoreCell calcScore(Movement m, Place p) {
        int placeId = p.getId();
        double score = p.getPopularity();
        return new ScoreCell(placeId, score);
    }

    public void printAccuracy() {
        System.out.println("Popularity-based predictor accuracy:" + accuracy);
    }

}
