package data;

import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by chao on 4/29/15.
 */

public class CheckinDataset {

    int N;
    int V;
    List<RealVector> geoData;
    List<RealVector> temporalData;
    List<Map<Integer, Integer>> textData;
    List<Double> weights; // The weights are the popularities of the places.
    double weightedSum; // The weighted sum of the given data points.

    public void load(SequenceDataset hmmd) {
        this.geoData = hmmd.getGeoData();
        this.temporalData = hmmd.getTemporalData();
        this.textData = hmmd.getTextData();
        this.N = geoData.size();
        this.V = hmmd.numWords();
        initWeights();
    }

    private void initWeights() {
        weights = new ArrayList<Double>();
        for (int i=0; i<N; i++)
            weights.add(1.0);
        weightedSum = N;
    }

    public int numPlace() {
        return N;
    }

    public int numWord() {
        return V;
    }

    public List<RealVector> getGeoData() {
        return geoData;
    }

    public List<RealVector> getTemporalData() {
        return temporalData;
    }

    public List<Map<Integer, Integer>> getTextData() {
        return textData;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public double getWeightedSum() {
        return weightedSum;
    }

    public RealVector getGeoDatum(int index) {
        return geoData.get(index);
    }

    public RealVector getTemporalDatum(int index) {
        return temporalData.get(index);
    }

    public Map<Integer, Integer> getTextDatum(int index) {
        return textData.get(index);
    }

    public double getWeight(int index) {
        return weights.get(index);
    }

}
