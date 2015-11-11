package outlier;

import data.Movement;
import data.MovementDB;
import data.Place;
import model.Background;
import model.Mixture;
import myutils.ScoreCell;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by chao on 5/4/15.
 */
public class HMMDetector extends OutlierDetecter {

    Mixture model;

    public HMMDetector(Mixture model) {
        this.model = model;
    }

    public ScoreCell calcScore(MovementDB mdb, Movement m, int movementId) {
        List<RealVector> geo = new ArrayList<RealVector>();
        List<Map<Integer,Integer>> text = new ArrayList<Map<Integer, Integer>>();
        Place startPlace = m.getStartPlace();
        Place endPlace = m.getEndPlace();
        geo.add(startPlace.getLocation().toRealVector());
        text.add(startPlace.getDescriptions());
        geo.add(endPlace.getLocation().toRealVector());
        text.add(endPlace.getDescriptions());
        double hmmLL = model.calcLL(geo, text);
        Background b = model.getBackgroundModel();
        double backLL = b.calcLL(geo.get(0), text.get(0));
        backLL += b.calcLL(geo.get(1), text.get(1));
        double score = backLL - hmmLL;
        return new ScoreCell(movementId, score);
    }

}
