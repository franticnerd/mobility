package model;

import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import data.HMMDatabase;
import myutils.ArrayUtils;
import org.apache.commons.math3.linear.RealVector;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    public Background getBackgroundModel() {
        return backgroundModel;
    }

    public void train(HMMDatabase data, int K, int M) {
        init(data, K, M);
        double prevLL = totalLL;
        for (int iter = 0; iter < maxIter; iter ++) {
            eStep(data);
            mStep(data);
            calcTotalLL(data);
            System.out.println("Mixture model finished iteration " + iter + ". Log-likelihood:" + totalLL);
            if(Math.abs(totalLL - prevLL) <= 0.1)
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
    protected void initMStepParameters(HMMDatabase data) {
        super.initMStepParameters(data);
        lambda = 0.5;
    }

    /**
     * Step 2: learning the parameters using EM: E-Step.
     */
    protected void eStep(HMMDatabase data) {
        super.eStep(data);
        calcKappa(data);
    }

    protected void calcKappa(HMMDatabase data) {
        for (int r=0; r<R; r++) {
            // Background LL
            kappa[r][0]= log(1.0 - lambda);
            for (int n=0; n<2; n++) {
                RealVector geoDatum = data.getGeoDatum(2*r+n);
                Map<Integer, Integer> textDatum = data.getTextDatum(2*r+n);
                kappa[r][0] += backgroundModel.calcLL(geoDatum, textDatum);
            }
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
    protected void mStep(HMMDatabase data) {
        updateLambda();
        updatePi();
        updateA();
        updateTextModel(data);
        updateGeoModel(data);
    }

    protected void updateLambda() {
        lambda = 0;
        for(int r=0; r<R; r++) {
            lambda += kappa[r][1];
        }
        lambda /= R;
    }

    protected void updatePi() {
        double denominator = 0;
        for (int r=0; r<R; r++) {
            denominator += kappa[r][1];
        }
        for(int k=0; k<K; k++) {
            double numerator = 0;
            for(int r=0; r<R; r++) {
                numerator += kappa[r][1] * gamma[r][0][k];
            }
            pi[k] = numerator / denominator;
        }
    }

    protected void updateA() {
        for (int j=0; j<K; j++) {
            double denominator = 0;
            for (int r=0; r<R; r++)
                for (int k=0; k<K; k++)
                    denominator += kappa[r][1] * xi[r][j][k];
            for (int k=0; k<K; k++) {
                double numerator = 0;
                for (int r=0; r<R; r++) {
                    numerator += kappa[r][1] * xi[r][j][k];
                }
                A[j][k] = numerator / denominator;
            }
        }
    }

    protected void updateTextModel(HMMDatabase data) {
        for(int k=0; k<K; k++) {
            List<Double> textWeights = new ArrayList<Double>();
            for (int r=0; r<R; r++)
                for (int n=0; n<2; n++)
                    textWeights.add(kappa[r][1]*gamma[r][n][k]);
            textModel[k].fit(V, data.getTextData(), textWeights);
        }
    }

    protected void updateGeoModel(HMMDatabase data) {
        updateC();
        for(int k=0; k<K; k++) {
            for (int m=0; m<M; m++) {
                List<Double> weights = new ArrayList<Double>();
                for (int r=0; r<R; r++)
                    for (int n=0; n<2; n++)
                        weights.add(kappa[r][1]*rho[r][n][k][m]);
                geoModel[k][m].fit(data.getGeoData(), weights);
            }
        }
    }

    protected void updateC() {
        for (int k=0; k<K; k++) {
            double denominator = 0;
            for (int r = 0; r < R; r++)
                for (int n = 0; n < 2; n++)
                    denominator += kappa[r][1] * gamma[r][n][k];
            for (int m = 0; m < M; m++) {
                double numerator = 0;
                for (int r = 0; r < R; r++)
                    for (int n = 0; n < 2; n++)
                        numerator += kappa[r][1] * rho[r][n][k][m];
                c[k][m] = numerator / denominator;
            }
        }
    }

    /**
     *
     * Functions for computing probabilities
     */

    protected void calcTotalLL(HMMDatabase data) {
        totalLL = 0;
        for (int r=0; r<R; r++) {
            double hmmLL = 0;
            for (int n=0; n<2; n++)
                hmmLL += con[r][n] + scalingFactor[r][n];
            double backgroundLL = 0;
            for (int n=0; n<2; n++) {
                double prob = backgroundModel.calcLL(data.getGeoDatum(2 * r + n), data.getTextDatum(2 * r + n));
                backgroundLL += prob;
            }
            double mixtureProb = lambda * exp(hmmLL) + (1 - lambda) * exp(backgroundLL);
            totalLL += log(mixtureProb);
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
//        o.put("kappa", kappa);
        o.put("lambda", lambda);
        o.put("background", backgroundModel.toBson());
        return o;
    }


    public void load(DBObject o) {
        super.load(o);
        this.lambda = (Double) o.get("lambda");

//        Object[] kappaList = ((BasicDBList)o.get("kappa")).toArray();
//        this.kappa = new double[kappaList.length][((BasicDBList)kappaList[0]).size()];
//        for(int i=0; i<kappaList.length; i++) {
//            BasicDBList list = (BasicDBList) kappaList[i];
//            for (int j = 0; j < list.size(); j++)
//                kappa[i][j] = (Double) list.get(j);
//        }

        DBObject bgd = (BasicDBObject) o.get("background");
        this.backgroundModel = new Background(bgd);
    }


}


//    protected boolean checkIsNaN() {
//        boolean isNaN = false;
//        for (int r=0; r<R; r++)
//            for (int n=0; n<2; n++)
//                for (int k=0; k<K; k++) {
//                    if (Double.isNaN(ll[r][n][k])) {
//                        isNaN = true;
//                        System.out.println("ll[r][n][k] is not a number!" + r + " " + n + " " + k + "\n");
//                    }
//                    if (Double.isNaN(alpha[r][n][k])) {
//                        isNaN = true;
//                        System.out.println("alpha[r][n][k] is not a number!" + r + " " + n + " " + k + "\n");
//                    }
//                    if (Double.isNaN(beta[r][n][k])) {
//                        isNaN = true;
//                        System.out.println("beta[r][n][k] is not a number!" + r + " " + n + " " + k + "\n");
//                    }
//                    if (Double.isNaN(gamma[r][n][k])) {
//                        isNaN = true;
//                        System.out.println("gamma[r][n][k] is not a number!" + r + " " + n + " " + k + "\n");
//                    }
//                }
//        for (int r=0; r<R; r++)
//            for (int j=0; j<K; j++)
//                for (int k=0; k<K; k++)
//                    if (Double.isNaN(xi[r][j][k])) {
//                        isNaN = true;
//                        System.out.println("xi[r][j][k] is not a number!" + r + " " + j + " " + k + "\n");
//                    }
//        for (int r=0; r<R; r++)
//            for (int n=0; n<2; n++) {
//                if (Double.isNaN(con[r][n])) {
//                    isNaN = true;
//                    System.out.println("con[r][n] is not a number!" + r + " " + n + "\n");
//                }
//                if (Double.isNaN(scalingFactor[r][n])) {
//                    isNaN = true;
//                    System.out.println("scalingFactor[r][n] is not a number!" + r + " " + n + "\n");
//                }
//            }
//        for (int r=0; r<R; r++)
//            for (int n=0; n<2; n++)
//                for (int k=0; k<K; k++)
//                    for (int m=0; m<M; m++) {
//                        if (Double.isNaN(rho[r][n][k][m])) {
//                            isNaN = true;
//                            System.out.println("rho[r][n][k] is not a number!" + r + " " + n + " " + k + " " + m + "\n");
//                        }
//                    }
//        return isNaN;
//    }
