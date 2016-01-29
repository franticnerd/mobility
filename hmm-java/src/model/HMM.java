package model;

import cluster.KMeans;
import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;

import data.Checkin;
import data.Sequence;
import data.SequenceDataset;
import data.WordDataset;
import distribution.Gaussian;
import distribution.Multinomial;
import myutils.ArrayUtils;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * The hidden Markov Model. Created by chao on 4/14/15.
 */
public class HMM implements Serializable {

	public HMM() {
	}

	public HMM(DBObject o) {
		load(o);
	}

	// Fixed parameters.
	int maxIter;
	int R; // The number of sequences.
	int K; // The number of latent states.
	int M; // The number of Gaussian components in each state.
	int V; // The number of words.
	// weight of the sequences.
	double[] weight;
	double weightSum;
	// The latent variables
	double[][][] alpha; // alpha[r][n][k] is for the n-th position of sequence r at state k.
	double[][][] beta; // beta[r][n][k] is for the n-th position of sequence r at state k.
	double[][] con; // con[r][n] is ln p(x_n | x_1, x_2, ... x_n-1), this is used for normalization.
	double[][][] gamma; // gamma[r][n][k]: probability that the n-th position of sequence r is at state k.
	double[][][] xi; // xi[r][j][k] is the probability that the 1st position of sequence r is state j and the 2nd position is k.
	double[][][][] rho; // rho[r][n][k][l]: probability that the n-th position of sequence r is at state k and from Gaussian component l.
	// The parameters that need to be inferred.
	double[] pi; // The prior distribution over the latent states.
	double[][] A; // The transition matrix for the K latent states.
	Multinomial[] textModel; // The K multinomial models for the text part.
	Gaussian[][] geoModel; // The K Gaussian mixtures for the latent states, each mixture has M components.
	Gaussian[] temporalModel;
	double[][] c; // c[k][m] is the probability of choosing component m for state k;
	// Log likelihood.
	double[][][] ll; // ll[r][n][k] is the log-likelihood p(x[r][n]|k).
	double[][] scalingFactor; // scalingFactor[r][n] is chosen from ll[r][n].
	double totalLL;

	public HMM(int maxIter) {
		this.maxIter = maxIter;
	}

	public void train(SequenceDataset data, int K, int M) {
		init(data, K, M);
		iterate(data);
	}

	/**
	 * Step 1: initialize the geo and text models.
	 */

	protected void init(SequenceDataset data, int K, int M) {
		init(data, K, M, null);
	}

	protected void init(SequenceDataset data, int K, int M, double[] weight) {
		initFixedParameters(data, K, M, weight);
		initEStepParameters();
		initMStepParameters(data);
	}

	protected void initFixedParameters(SequenceDataset data, int K, int M, double[] weight) {
		this.R = data.size();
		this.K = K;
		this.M = M;
		this.V = data.numWords();
		setWeight(weight);
	}

	private void setWeight(double[] weight) {
		if (weight == null) {
			this.weight = new double[R];
			for (int i = 0; i < R; i++)
				this.weight[i] = 1.0;
		} else {
			this.weight = Arrays.copyOf(weight, weight.length);
		}
		weightSum = ArrayUtils.sum(this.weight);
	}

	protected void initEStepParameters() {
		ll = new double[R][2][K];
		scalingFactor = new double[R][2];
		alpha = new double[R][2][K];
		beta = new double[R][2][K];
		con = new double[R][2];
		xi = new double[R][K][K];
		gamma = new double[R][2][K];
		rho = new double[R][2][K][M];
	}

	// Initialize the paramters that need to be inferred.
	protected void initMStepParameters(SequenceDataset data) {
		List<Integer>[] kMeansResults = runKMeans(data);
		initPi(kMeansResults);
		initA(kMeansResults);
		initTextModel(kMeansResults, data);
		initGeoModel(kMeansResults, data);
		initTemporalModel(kMeansResults, data);
	}

	// Run k-means for the geo data to initialize the params.
	protected List<Integer>[] runKMeans(SequenceDataset data) {
		List<Double> weights = new ArrayList<Double>();
		for (int i = 0; i < data.getGeoData().size(); i++)
			weights.add(weight[i / 2]);
		KMeans kMeans = new KMeans(500);
		return kMeans.cluster(data.getGeoData(), data.getTemporalData(), weights, K);
	}

	// numDataPoints is 2R.
	protected void initPi(List<Integer>[] kMeansResults) {
		pi = new double[K];
		for (int i = 0; i < K; i++) {
			List<Integer> members = kMeansResults[i];
			double numerator = 0;
			for (Integer m : members) {
				numerator += weight[m / 2];
			}
			pi[i] = numerator / (2 * weightSum);
		}
	}

	protected void initA(List<Integer>[] kMeansResults) {
		A = new double[K][K];
		int[] dataMembership = findMemebership(kMeansResults);
		for (int r = 0; r < R; r++) {
			int fromClusterId = dataMembership[2 * r];
			int toClusterId = dataMembership[2 * r + 1];
			A[fromClusterId][toClusterId] += weight[r];
		}
		for (int i = 0; i < K; i++) {
			double rowSum = ArrayUtils.sum(A[i]);
			if (rowSum == 0) {
				System.out.println("Transition matrix row is all zeros." + i);
			}
			for (int j = 0; j < K; j++) {
				A[i][j] /= rowSum;
			}
		}
	}

	// Find the kmeans membership for the 2*R places
	protected int[] findMemebership(List<Integer>[] kMeansResults) {
		int[] dataMembership = new int[2 * R];
		for (int clusterId = 0; clusterId < K; clusterId++) {
			List<Integer> clusterDataIds = kMeansResults[clusterId];
			for (int dataId : clusterDataIds)
				dataMembership[dataId] = clusterId;
		}
		return dataMembership;
	}

	protected void initTextModel(List<Integer>[] kMeansResults, SequenceDataset data) {
		this.textModel = new Multinomial[K];
		for (int i = 0; i < K; i++) {
			List<Integer> dataIds = kMeansResults[i];
			List<Map<Integer, Integer>> clusterData = new ArrayList<Map<Integer, Integer>>();
			List<Double> clusterWeights = new ArrayList<Double>();
			for (int dataId : dataIds) {
				clusterData.add(data.getTextData().get(dataId));
				clusterWeights.add(weight[dataId / 2]);
			}
			textModel[i] = new Multinomial();
			textModel[i].fit(V, clusterData, clusterWeights);
		}
	}

	// Initialize the geo model and c
	protected void initGeoModel(List<Integer>[] kMeansResults, SequenceDataset data) {
		this.geoModel = new Gaussian[K][M]; // K states, each having M components
		this.c = new double[K][M];
		for (int k = 0; k < K; k++) {
			List<RealVector> clusterData = new ArrayList<RealVector>();
			List<Double> clusterWeights = new ArrayList<Double>();
			List<Integer> dataIds = kMeansResults[k];
			for (int dataId : dataIds) {
				clusterData.add(data.getGeoDatum(dataId));
				clusterWeights.add(weight[dataId / 2]);
			}
			KMeans kMeans = new KMeans(500);
			List<Integer>[] subKMeansResults = kMeans.cluster(clusterData, clusterWeights, M);
			for (int m = 0; m < M; m++) {
				List<Integer> subDataIds = subKMeansResults[m];
				List<RealVector> subClusterData = new ArrayList<RealVector>();
				List<Double> subClusterWeights = new ArrayList<Double>();
				for (int dataId : subDataIds) {
					subClusterData.add(clusterData.get(dataId));
					subClusterWeights.add(clusterWeights.get(dataId));
				}
				geoModel[k][m] = new Gaussian();
				geoModel[k][m].fit(subClusterData, subClusterWeights);
				c[k][m] = ArrayUtils.sum(subClusterWeights) / ArrayUtils.sum(clusterWeights);
			}
		}
	}

	// Initialize the temporal model and c
	protected void initTemporalModel(List<Integer>[] kMeansResults, SequenceDataset data) {
		this.temporalModel = new Gaussian[K]; // K states, each having M components
		for (int k = 0; k < K; k++) {
			List<RealVector> clusterData = new ArrayList<RealVector>();
			List<Double> clusterWeights = new ArrayList<Double>();
			List<Integer> dataIds = kMeansResults[k];
			for (int dataId : dataIds) {
				clusterData.add(data.getTemporalDatum(dataId));
				clusterWeights.add(weight[dataId / 2]);
			}
			temporalModel[k] = new Gaussian();
			temporalModel[k].fit(clusterData, clusterWeights);
		}
	}

	/**
	 * Step 2: iterate over the e-step and m-step.
	 */
	protected void iterate(SequenceDataset data) {
		double prevLL = totalLL;
		for (int iter = 0; iter < maxIter; iter++) {
			eStep(data);
			mStep(data);
			calcTotalLL();
//			System.out.println("HMM finished iteration " + iter + ". Log-likelihood:" + totalLL);
			if (Math.abs(totalLL - prevLL) <= 0.01)
				break;
			prevLL = totalLL;
		}
	}

	/**
	 * Step 2.1: learning the parameters using EM: E-Step.
	 */
	protected void eStep(SequenceDataset data) {
		calcLL(data);
		scaleLL();
		calcAlpha();
		calcBeta();
		calcGamma();
		calcXi();
		calcRho(data);
	}

	// Compute the log likelihood.
	protected void calcLL(SequenceDataset data) {
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++)
					ll[r][n][k] = calcLLState(data.getGeoDatum(2 * r + n), data.getTemporalDatum(2 * r + n),
							data.getTextDatum(2 * r + n), k);
	}

	protected void scaleLL() {
		// Find the scaling factors.
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				scalingFactor[r][n] = ArrayUtils.max(ll[r][n]);
		// Scale the log-likelihood.
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++)
					ll[r][n][k] -= scalingFactor[r][n];
	}

	protected void calcAlpha() {
		// Compute alpha[r][0][k], in the log domain!
		for (int r = 0; r < R; r++)
			for (int k = 0; k < K; k++)
				alpha[r][0][k] = log(pi[k]) + ll[r][0][k];
		// Compute con[r][0], namely ln p(x_0)
		for (int r = 0; r < R; r++)
			con[r][0] = ArrayUtils.sumExpLog(alpha[r][0]);
		// Normalize alpha[r][0][k]
		for (int r = 0; r < R; r++)
			ArrayUtils.logNormalize(alpha[r][0]);
		// Compute alpha[r][1][k], again in the log domain.
		for (int r = 0; r < R; r++) {
			for (int k = 0; k < K; k++) {
				alpha[r][1][k] = ll[r][1][k];
				double sum = 1e-200;
				for (int j = 0; j < K; j++) {
					sum += alpha[r][0][j] * A[j][k];
				}
				alpha[r][1][k] += log(sum);
			}
		}
		// Compute con[r][1], namely ln p(x_1 | x_0)
		for (int r = 0; r < R; r++)
			con[r][1] = ArrayUtils.sumExpLog(alpha[r][1]);
		// Normalize alpha[r][1][k]
		for (int r = 0; r < R; r++)
			ArrayUtils.logNormalize(alpha[r][1]);
	}

	protected void calcBeta() {
		// Compute beta[r][1][k]
		for (int r = 0; r < R; r++)
			for (int k = 0; k < K; k++)
				beta[r][1][k] = 1.0;
		// Compute beta[r][0][k]
		for (int r = 0; r < R; r++) {
			for (int k = 0; k < K; k++) {
				double sum = 0;
				for (int j = 0; j < K; j++) {
					if (A[k][j] == 0)
						sum += 0;
					else if (ll[r][1][j] - con[r][1] >= 500)
						sum += A[k][j] * 1e200;
					else
						sum += exp(ll[r][1][j] - con[r][1]) * A[k][j];
				}
				beta[r][0][k] = sum;
			}
		}
	}

	protected void calcGamma() {
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++)
					gamma[r][n][k] = alpha[r][n][k] * beta[r][n][k];
	}

	protected void calcXi() {
		for (int r = 0; r < R; r++)
			for (int j = 0; j < K; j++)
				for (int k = 0; k < K; k++)
					xi[r][j][k] = alpha[r][0][j] * exp(ll[r][1][k] - con[r][1]) * A[j][k] * beta[r][1][k];
	}

	protected void calcRho(SequenceDataset data) {
		for (int r = 0; r < R; r++) {
			for (int n = 0; n < 2; n++) {
				RealVector v = data.getGeoDatum(2 * r + n);
				for (int k = 0; k < K; k++) {
					for (int m = 0; m < M; m++)
						rho[r][n][k][m] = calcGeoLLComponent(v, k, m); // Log domain.
					ArrayUtils.logNormalize(rho[r][n][k]); // Transform to normal domain.
					for (int m = 0; m < M; m++)
						rho[r][n][k][m] = gamma[r][n][k] * rho[r][n][k][m];
				}
			}
		}
	}

	/**
	 * Step 2.2: learning the parameters using EM: M-Step.
	 */
	protected void mStep(SequenceDataset data) {
		updatePi();
		updateA();
		updateTextModel(data);
		updateGeoModel(data);
		updateTemporalModel(data);
	}

	protected void updatePi() {
		for (int k = 0; k < K; k++) {
			double numerator = 0;
			for (int r = 0; r < R; r++) {
				numerator += gamma[r][0][k] * weight[r];
			}
			pi[k] = numerator / weightSum;
		}
	}

	protected void updateA() {
		for (int j = 0; j < K; j++) {
			double denominator = 0;
			for (int r = 0; r < R; r++)
				for (int k = 0; k < K; k++)
					denominator += weight[r] * xi[r][j][k];
			for (int k = 0; k < K; k++) {
				double numerator = 0;
				for (int r = 0; r < R; r++) {
					numerator += weight[r] * xi[r][j][k];
				}
				A[j][k] = numerator / denominator;
			}
		}
	}

	protected void updateTextModel(SequenceDataset data) {
		for (int k = 0; k < K; k++) {
			List<Double> textWeights = new ArrayList<Double>();
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++)
					textWeights.add(weight[r] * gamma[r][n][k]);
			textModel[k].fit(V, data.getTextData(), textWeights);
		}
	}

	protected void updateGeoModel(SequenceDataset data) {
		updateC();
		for (int k = 0; k < K; k++) {
			for (int m = 0; m < M; m++) {
				List<Double> weights = new ArrayList<Double>();
				for (int r = 0; r < R; r++)
					for (int n = 0; n < 2; n++)
						weights.add(weight[r] * rho[r][n][k][m]);
				geoModel[k][m].fit(data.getGeoData(), weights);
			}
		}
	}

	protected void updateC() {
		for (int k = 0; k < K; k++) {
			double denominator = 0;
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++)
					denominator += weight[r] * gamma[r][n][k];
			for (int m = 0; m < M; m++) {
				double numerator = 0;
				for (int r = 0; r < R; r++)
					for (int n = 0; n < 2; n++)
						numerator += weight[r] * rho[r][n][k][m];
				c[k][m] = numerator / denominator;
			}
		}
	}

	protected void updateTemporalModel(SequenceDataset data) {
		for (int k = 0; k < K; k++) {
			List<Double> weights = new ArrayList<Double>();
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++)
					weights.add(weight[r] * gamma[r][n][k]);
			temporalModel[k].fit(data.getTemporalData(), weights);
		}
	}

	/**
	 * Functions for computing probabilities
	 */
	// Calc the likelihood that the given data is generated from state k
	protected double calcLLState(RealVector geoDatum, RealVector temporalDatum, Map<Integer, Integer> textDatum,
			int k) {
		return calcLLState(geoDatum, temporalDatum, textDatum, k, false);
	}

	protected double calcLLState(RealVector geoDatum, RealVector temporalDatum, Map<Integer, Integer> textDatum, int k,
			boolean isTest) {
		double textProb = textModel[k].calcLL(textDatum, isTest);
		double geoProb = calcGeoLLState(geoDatum, k);
		double temporalProb = temporalModel[k].calcLL(temporalDatum);
		return geoProb + temporalProb + textProb;
	}

	// Calc the probability that v is generated from the gmm of state k.
	protected double calcGeoLLState(RealVector v, int k) {
		double[] lnProb = new double[M];
		for (int m = 0; m < M; m++)
			lnProb[m] = calcGeoLLComponent(v, k, m);
		double maxLnProb = ArrayUtils.max(lnProb);
		for (int m = 0; m < M; m++)
			lnProb[m] -= maxLnProb;
		double sum = 0;
		for (int m = 0; m < M; m++)
			sum += exp(lnProb[m]);
		return maxLnProb + log(sum);
	}

	// Compute the prob that v is generated from the m-th component of state k.
	protected double calcGeoLLComponent(RealVector v, int k, int m) {
		double prior = c[k][m];
		double logGeoProb = geoModel[k][m].calcLL(v);
		return log(prior) + logGeoProb;
	}

	protected void calcTotalLL() {
		totalLL = 0;
		for (int r = 0; r < R; r++) {
			double hmmLL = 0;
			for (int n = 0; n < 2; n++)
				hmmLL += con[r][n] + scalingFactor[r][n];
			totalLL += weight[r] * hmmLL;
		}
	}

	/**
	 * Functions for output.
	 */
	public String toString(WordDataset wd) {

		// Write K M.
		String s = "# K M\n";
		s += K + " " + M + "\n";

		// Write Pi.
		s += "# Pi\n";
		for (int i = 0; i < K; i++)
			s += pi[i] + " ";
		s += "\n";

		// Write A.
		s += "# Transition\n";
		for (int j = 0; j < K; j++) {
			for (int k = 0; k < K; k++) {
				s += A[j][k] + " ";
			}
			s += "\n";
		}

		// Write geo model.
		s += "# geo\n";
		for (int k = 0; k < K; k++) {
			for (int m = 0; m < M; m++) {
				s += k + "," + m + "," + pi[k] + "," + c[k][m] + ",";
				RealVector mean = geoModel[k][m].getMean();
				s += mean.getEntry(0) + "," + mean.getEntry(1) + ",";
				RealMatrix var = geoModel[k][m].getVar();
				s += var.getEntry(0, 0) + "," + var.getEntry(0, 1) + "," + var.getEntry(1, 0) + "," + var.getEntry(1, 1)
						+ "\n";
			}
		}

		// write temporal model.
		s += "# temporal\n";
		for (int i = 0; i < K; i++) {
			RealVector mean = temporalModel[i].getMean();
			RealMatrix var = temporalModel[i].getVar();
			s += mean.getEntry(0) + " ";
			s += var.getEntry(0, 0) + "\n";
		}

		// Write text model.
		s += "# text\n";
		for (int i = 0; i < K; i++) {
			s += "------------------------------ State " + i + "------------------------------\n";
			s += textModel[i].getWordDistribution(wd, 20) + "\n"; // Output the top 20 words.
		}
		return s;
	}

	// Load from a model file.
	public static HMM load(String inputFile) throws Exception {
		ObjectInputStream objectinputstream = new ObjectInputStream(new FileInputStream(inputFile));
		HMM m = (HMM) objectinputstream.readObject();
		objectinputstream.close();
		return m;
	}

	// Serialize
	public void serialize(String serializeFile) throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serializeFile));
		oos.writeObject(this);
		oos.close();
	}

	public void write(WordDataset wd, String outputFile) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile, false));
		bw.append(this.toString(wd));
		bw.close();
	}

	/**
	 * Compute the ll of a test sequence.
	 */
	public double calcLL(List<RealVector> geo, List<RealVector> temporal, List<Map<Integer, Integer>> text) {
		return calcLL(geo, temporal, text, false);
	}

	public double calcLL(List<RealVector> geo, List<RealVector> temporal, List<Map<Integer, Integer>> text,
			boolean isTest) {
		double[][] ll = new double[2][K]; // ll[n][k] is the log-likelihood p(x[n]|k).
		double[] scalingFactor = new double[2]; // scalingFactor[n] is chosen from ll[n].
		double[][] alpha = new double[2][K]; // alpha[n][k] is for the n-th position of sequence r at state k.
		double[] con = new double[2]; // con[n] is ln p(x_n | x_1, x_2, ... x_n-1), this is used for normalization.
		// calc LL
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] = calcLLState(geo.get(n), temporal.get(n), text.get(n), k, isTest);
		// Find the scaling factors.
		for (int n = 0; n < 2; n++)
			scalingFactor[n] = ArrayUtils.max(ll[n]);
		// Scale the log-likelihood.
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] -= scalingFactor[n];
		// Compute alpha[0][k], in the log domain!
		for (int k = 0; k < K; k++)
			alpha[0][k] = log(pi[k]) + ll[0][k];
		// Compute con[0], namely ln p(x_0)
		con[0] = ArrayUtils.sumExpLog(alpha[0]);
		// Normalize alpha[0][k]
		ArrayUtils.logNormalize(alpha[0]);
		// Compute alpha[1][k], again in the log domain.
		for (int k = 0; k < K; k++) {
			alpha[1][k] = ll[1][k];
			double sum = 1e-200;
			for (int j = 0; j < K; j++) {
				sum += alpha[0][j] * A[j][k];
			}
			alpha[1][k] += log(sum);
		}
		// Compute con[1], namely ln p(x_1 | x_0)
		con[1] = ArrayUtils.sumExpLog(alpha[1]);
		// the result ll.
		double hmmLL = 0;
		for (int n = 0; n < 2; n++)
			hmmLL += con[n] + scalingFactor[n];
		return hmmLL;
	}

	public double calcSeqScore(Sequence seq) {
		Checkin startPlace = seq.getCheckin(0);
		Checkin endPlace = seq.getCheckin(1);
		List<RealVector> geo = new ArrayList<RealVector>();
		List<RealVector> temporal = new ArrayList<RealVector>();
		List<Map<Integer, Integer>> text = new ArrayList<Map<Integer, Integer>>();
		geo.add(startPlace.getLocation().toRealVector());
		temporal.add(new ArrayRealVector(new double[] { startPlace.getTimestamp() % 1440 }));
		text.add(startPlace.getMessage());
		geo.add(endPlace.getLocation().toRealVector());
		temporal.add(new ArrayRealVector(new double[] { endPlace.getTimestamp() % 1440 }));
		text.add(endPlace.getMessage());
		return calcLL(geo, temporal, text);
	}

	public DBObject toBson() {
		DBObject o = new BasicDBObject();
		o.put("R", R);
		o.put("K", K);
		o.put("M", M);
		o.put("V", V);
		o.put("pi", pi);
		o.put("A", A);
		o.put("c", c);
		List<DBObject> text = new ArrayList<DBObject>();
		for (Multinomial m : textModel)
			text.add(m.toBSon());
		o.put("textModel", text);
		List<List<DBObject>> geo = new ArrayList<List<DBObject>>();
		for (int i = 0; i < geoModel.length; i++) {
			List<DBObject> gmmdata = new ArrayList<DBObject>();
			Gaussian[] gmm = geoModel[i];
			for (int j = 0; j < gmm.length; j++) {
				gmmdata.add(gmm[j].toBSon());
			}
			geo.add(gmmdata);
		}
		o.put("geoModel", geo);
		List<DBObject> temporal = new ArrayList<DBObject>();
		for (Gaussian t : temporalModel)
			temporal.add(t.toBSon());
		o.put("temporalModel", temporal);
		return o;
	}


	public DBObject statsToBson() {
		DBObject o = new BasicDBObject();
		o.put("maxIter", maxIter);
		o.put("R", R);
		o.put("K", K);
		o.put("M", M);
		o.put("V", V);
		return o;
	}

	public void load(DBObject o) {
		this.R = (Integer) o.get("R");
		this.K = (Integer) o.get("K");
		this.M = (Integer) o.get("M");
		this.V = (Integer) o.get("V");

		BasicDBList piList = (BasicDBList) o.get("pi");
		this.pi = new double[piList.size()];
		for (int i = 0; i < piList.size(); i++) {
			this.pi[i] = (Double) piList.get(i);
		}

		Object[] aList = ((BasicDBList) o.get("A")).toArray();
		this.A = new double[aList.length][((BasicDBList) aList[0]).size()];
		for (int i = 0; i < aList.length; i++) {
			BasicDBList list = (BasicDBList) aList[i];
			for (int j = 0; j < list.size(); j++)
				A[i][j] = (Double) list.get(j);
		}

		Object[] cList = ((BasicDBList) o.get("c")).toArray();
		this.c = new double[cList.length][((BasicDBList) cList[0]).size()];
		for (int i = 0; i < cList.length; i++) {
			BasicDBList list = (BasicDBList) cList[i];
			for (int j = 0; j < list.size(); j++)
				c[i][j] = (Double) list.get(j);
		}

		List<DBObject> text = (List<DBObject>) o.get("textModel");
		this.textModel = new Multinomial[text.size()];
		for (int i = 0; i < text.size(); i++)
			this.textModel[i] = new Multinomial(text.get(i));

		List<List<DBObject>> geo = (List<List<DBObject>>) o.get("geoModel");
		int row = geo.size(), column = geo.get(0).size();
		geoModel = new Gaussian[row][column];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				DBObject d = geo.get(i).get(j);
				Gaussian g = new Gaussian(d);
				geoModel[i][j] = g;
			}
		}

		List<DBObject> temporal = (List<DBObject>) o.get("temporalModel");
		this.temporalModel = new Gaussian[text.size()];
		for (int i = 0; i < temporal.size(); i++)
			this.temporalModel[i] = new Gaussian(temporal.get(i));
	}

	/**
	 * Methods for the ensemble of HMM.
	 */
	// don't need to return LL, since I noticed the LL is stored in "totalLL" and can be accessed any time
	public void train(SequenceDataset data, int K, int M, double[] seqsFracCount) {
		init(data, K, M, seqsFracCount);
		iterate(data);
	}

	// don't need to return LL, since I noticed the LL is stored in "totalLL" and can be accessed any time
	public void update(SequenceDataset data, double[] seqsFracCount) {
		setWeight(seqsFracCount);
		iterate(data);
	}

	public double getTotalLL() {
		return totalLL;
	}

}
