package predict;

import data.Checkin;
import data.Sequence;
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

    public ScoreCell calcScore(Sequence m, Checkin p) {
        Checkin startPlace = m.getCheckin(0);
        List<RealVector> geo = new ArrayList<RealVector>();
        List<Map<Integer,Integer>> text = new ArrayList<Map<Integer, Integer>>();
        geo.add(startPlace.getLocation().toRealVector());
        text.add(startPlace.getMessage());
        geo.add(p.getLocation().toRealVector());
        text.add(p.getMessage());
        double score = model.calcLL(geo, text);
        int placeId = p.getId();
        return new ScoreCell(placeId, score);
    }

    public void printAccuracy() {
        System.out.println("HMM-based predictor accuracy:" + accuracy);
    }

}
