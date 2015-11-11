package predict;

import data.Movement;
import data.Place;
import myutils.ScoreCell;

/**
 * Created by chao on 5/3/15.
 */
public class DistancePredictor extends Predictor {

    public ScoreCell calcScore(Movement m, Place p) {
        int placeId = p.getId();
        Place startPlace = m.getStartPlace();
        double score = p.getGeographicDist(startPlace);
        return new ScoreCell(placeId, score);
    }

    public void printAccuracy() {
        System.out.println("Distance-based predictor accuracy:" + accuracy);
    }

}
