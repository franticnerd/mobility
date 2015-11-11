package data;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by chao on 4/29/15.
 */

public class BackgroundDatabase {

    int N;
    int V;
    List<RealVector> geoData;
    List<Map<Integer, Integer>> textData;
    List<Double> weights; // The weights are the popularities of the places.
    double weightedSum; // The weighted sum of the given data points.

    public void load() {
        this.N = Database.numPlace();
        this.V = Database.numWord();
        initWeights();  // Initialize the weights as the popularity of the places.
        initGeoData();  // Initialize the geo data.
        initTextData();  // Initialize the geo data.
    }

    private void initWeights() {
        weights = new ArrayList<Double>();
        for(int i=0; i<N; i++) {
            Place place = Database.getPlace(i);
            weights.add( (double)place.getPopularity() );
        }
        weightedSum = 0;
        for (int i=0; i<N; i++)
            weightedSum += weights.get(i);
    }

    private void initGeoData() {
        geoData = new ArrayList<RealVector>();
        for(int i=0; i<N; i++) {
            Place place = Database.getPlace(i);
            double lng = place.getLocation().getLng();
            double lat = place.getLocation().getLat();
            geoData.add( new ArrayRealVector(new double[] {lng, lat}) );
        }
    }

    private void initTextData() {
        PlaceDatabase bgPd = Database.pd.copy();
        List<Checkin> checkins = Database.cd.getAllCheckins();
        for (Checkin ci : checkins) {
            int placeId = ci.getPlaceId();
            bgPd.getPlace(placeId).addMessage(ci.getMessage());
        }
        textData = new ArrayList<Map<Integer, Integer>>();
        for(int i=0; i<bgPd.size(); i++) {
            Place place = bgPd.getPlace(i);
            Map<Integer, Integer> text = place.getDescriptions();
            textData.add(text);
        }
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

    public Map<Integer, Integer> getTextDatum(int index) {
        return textData.get(index);
    }

    public double getWeight(int index) {
        return weights.get(index);
    }

}
