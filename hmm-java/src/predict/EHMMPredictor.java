package predict;

import data.Checkin;
import data.PredictionDataset;
import data.Sequence;
import model.HMM;
import model.EHMM;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class EHMMPredictor extends Predictor {

	EHMM model;
	boolean avgTest;

	public EHMMPredictor(EHMM model, boolean avgTest) {
		this.model = model;
		this.avgTest = avgTest;
	}

	public ScoreCell calcScore(Sequence m, Checkin p) {
		Checkin startPlace = m.getCheckin(0);
		List<RealVector> geo = new ArrayList<RealVector>();
		List<RealVector> temporal = new ArrayList<RealVector>();
		List<Map<Integer, Integer>> text = new ArrayList<Map<Integer, Integer>>();
		geo.add(startPlace.getLocation().toRealVector());
		temporal.add(new ArrayRealVector(new double[] { startPlace.getTimestamp() % 1440 }));
		text.add(startPlace.getMessage());
		geo.add(p.getLocation().toRealVector());
		temporal.add(new ArrayRealVector(new double[] { p.getTimestamp() % 1440 }));
		text.add(p.getMessage());
		double score = model.calcLL(m.getUserId(), geo, temporal, text, avgTest);
		int checkinId = p.getId();
		return new ScoreCell(checkinId, score);
	}

	public void printAccuracy() {
		System.out.println("EHMM-based predictor accuracy:" + accuracy);
	}

}
