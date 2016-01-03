package model;

import com.mongodb.DBObject;
import data.SequenceDataset;
import myutils.ArrayUtils;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * A mixture model that consists of background model and HMM.
 * Created by chao on 4/14/15.
 */
public class Mixture extends HMM {

    Background backgroundModel;
    // The latent variables
    double [][] kappa; // kappa[r][1] is the probability that sequence r is generated from HMM.
    // The parameters that need to be inferred.
    double lambda; // The probability of choosing HMM.

    public Mixture(int maxIter, Background background) {
        super(maxIter);
        this.backgroundModel = background;
    }

    public Mixture(DBObject o) {
        load(o);
    }

    public Mixture(DBObject background, DBObject o) {
        load(background, o);
    }

    public Background getBackgroundModel() {
        return backgroundModel;
    }

    public void train(SequenceDataset data, int K, int M) {
        init(data, K, M);
        double prevLL = totalLL;
        for (int iter = 0; iter < maxIter; iter ++) {
            eStep(data);
            mStep(data);
            calcTotalLL(data);
            System.out.println("Mixture model finished iteration " + iter + ". Log-likelihood:" + totalLL);
            if(Math.abs(totalLL - prevLL) <= 0.01)
                break;
            prevLL = totalLL;
        }
    }


    /**
     * Step 1: initialize.
     */
    protected void initEStepParameters() {
        super.initEStepParameters();
        kappa = new double [R][2];
    }

    // Initialize the paramters that need to be inferred.
    protected void initMStepParameters(SequenceDataset data) {
        super.initMStepParameters(data);
        lambda = 0.5;
    }

    /**
     * Step 2: learning the parameters using EM: E-Step.
     */
    protected void eStep(SequenceDataset data) {
        super.eStep(data);
        calcKappa(data);
    }

    protected void calcKappa(SequenceDataset data) {
        for (int r=0; r<R; r++) {
            // Background LL

//            kappa[r][0]= log(1.0 - lambda);
//            for (int n=0; n<2; n++) {
//                RealVector geoDatum = data.getGeoDatum(2*r+n);
//                Map<Integer, Integer> textDatum = data.getTextDatum(2*r+n);
//                kappa[r][0] += backgroundModel.calcLL(geoDatum, textDatum);
//            }

            kappa[r][0]= log(1.0 - lambda);
            kappa[r][0] += backgroundModel.calcLL(data.getGeoDatum(2*r), data.getTemporalDatum(2*r), data.getTextDatum(2*r),
                    data.getGeoDatum(2*r + 1), data.getTemporalDatum(2*r + 1), data.getTextDatum(2*r + 1));

            // HMM LL
            kappa[r][1]= log(lambda);
            // Note that p(X) has been scaled: p(X) = p(X^0) / exp(scale)
            for (int n=0; n<2; n++)
                kappa[r][1] += con[r][n] + scalingFactor[r][n];
            ArrayUtils.logNormalize(kappa[r]);
        }
    }


    /**
     * Step 3: learning the parameters using EM: M-Step.
     */
    protected void mStep(SequenceDataset data) {
        updateLambda();
        updatePi();
        updateA();
        updateTextModel(data);
        updateGeoModel(data);
        updateTemporalModel(data);
    }

    protected void updateLambda() {
        lambda = 0;
        for(int r=0; r<R; r++) {
            lambda += weight[r] * kappa[r][1];
        }
        lambda /= weightSum;
    }

    protected void updatePi() {
        double denominator = 0;
        for (int r=0; r<R; r++) {
            denominator += weight[r] * kappa[r][1];
        }
        for(int k=0; k<K; k++) {
            double numerator = 0;
            for(int r=0; r<R; r++) {
                numerator += weight[r] * kappa[r][1] * gamma[r][0][k];
            }
            pi[k] = numerator / denominator;
        }
    }

    protected void updateA() {
        for (int j=0; j<K; j++) {
            double denominator = 0;
            for (int r=0; r<R; r++)
                for (int k=0; k<K; k++)
                    denominator += weight[r] * kappa[r][1] * xi[r][j][k];
            for (int k=0; k<K; k++) {
                double numerator = 0;
                for (int r=0; r<R; r++) {
                    numerator += weight[r] * kappa[r][1] * xi[r][j][k];
                }
                A[j][k] = numerator / denominator;
            }
        }
    }

    protected void updateTextModel(SequenceDataset data) {
        for(int k=0; k<K; k++) {
            List<Double> textWeights = new ArrayList<Double>();
            for (int r=0; r<R; r++)
                for (int n=0; n<2; n++)
                    textWeights.add(kappa[r][1]*gamma[r][n][k]);
            textModel[k].fit(V, data.getTextData(), textWeights);
        }
    }

    protected void updateGeoModel(SequenceDataset data) {
        updateC();
        for(int k=0; k<K; k++) {
            for (int m=0; m<M; m++) {
                List<Double> weights = new ArrayList<Double>();
                for (int r=0; r<R; r++)
                    for (int n=0; n<2; n++)
                        weights.add(weight[r] * kappa[r][1]*rho[r][n][k][m]);
                geoModel[k][m].fit(data.getGeoData(), weights);
            }
        }
    }


    protected void updateTemporalModel(SequenceDataset data) {
        for(int k=0; k<K; k++) {
            List<Double> weights = new ArrayList<Double>();
            for (int r=0; r<R; r++)
                for (int n=0; n<2; n++)
                    weights.add(weight[r] * kappa[r][1]*gamma[r][n][k]);
            temporalModel[k].fit(data.getTemporalData(), weights);
        }
    }

    protected void updateC() {
        for (int k=0; k<K; k++) {
            double denominator = 0;
            for (int r = 0; r < R; r++)
                for (int n = 0; n < 2; n++)
                    denominator += weight[r] * kappa[r][1] * gamma[r][n][k];
            for (int m = 0; m < M; m++) {
                double numerator = 0;
                for (int r = 0; r < R; r++)
                    for (int n = 0; n < 2; n++)
                        numerator += weight[r] * kappa[r][1] * rho[r][n][k][m];
                c[k][m] = numerator / denominator;
            }
        }
    }

    /**
     *
     * Functions for computing probabilities
     */

    protected void calcTotalLL(SequenceDataset data) {
        totalLL = 0;
        for (int r=0; r<R; r++) {
            double hmmLL = 0;
            for (int n=0; n<2; n++)
                hmmLL += con[r][n] + scalingFactor[r][n];
            double backgroundLL = backgroundModel.calcLL(data.getGeoDatum(2 * r), data.getTemporalDatum(2*r), data.getTextDatum(2 * r),
                    data.getGeoDatum(2 * r + 1), data.getTemporalDatum(2 * r + 1), data.getTextDatum(2 * r + 1));
            double mixtureProb = lambda * exp(hmmLL) + (1 - lambda) * exp(backgroundLL);
            totalLL += weight[r] * log(mixtureProb);
        }
    }

    @Override
    public String toString() {
        String s = super.toString();
        return s + "# lambda:\n" + lambda + "\n";
    }

    // Load from a model file.
    public static Mixture load(String inputFile) throws Exception {
        ObjectInputStream objectinputstream = new ObjectInputStream(new FileInputStream(inputFile));
        Mixture m = (Mixture) objectinputstream.readObject();
        objectinputstream.close();
        return m;
    }

    // Serialize
    public void serialize(String serializeFile) throws Exception {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serializeFile));
        oos.writeObject(this);
        oos.close();
    }

    public DBObject toBson() {
        DBObject o = super.toBson();
        o.put("lambda", lambda);
        return o;
    }

    public void load(DBObject background, DBObject o) {
        super.load(o);
        this.lambda = (Double) o.get("lambda");
        this.backgroundModel = new Background(background);
    }

}


