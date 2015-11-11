package predict;

import data.Movement;
import data.Place;
import model.HMM;
import myutils.ScoreCell;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by chao on 5/2/15.
 */
public class HMMPredictor extends Predictor {

    HMM model;

    public HMMPredictor(HMM model) {
        this.model = model;
    }

    public ScoreCell calcScore(Movement m, Place p) {
        Place startPlace = m.getStartPlace();
        List<RealVector> geo = new ArrayList<RealVector>();
        List<Map<Integer,Integer>> text = new ArrayList<Map<Integer, Integer>>();
        geo.add(startPlace.getLocation().toRealVector());
        text.add(startPlace.getDescriptions());
        geo.add(p.getLocation().toRealVector());
        text.add(p.getDescriptions());
        double score = model.calcLL(geo, text);
        int placeId = p.getId();
        return new ScoreCell(placeId, score);
    }

    public void printAccuracy() {
        System.out.println("HMM-based predictor accuracy:" + accuracy);
    }

}
