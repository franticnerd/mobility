package model;

import java.util.*;
import myutils.*;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import cluster.KMeans;
import data.Checkin;
import data.CheckinDataset;
import data.Sequence;
import data.SequenceDataset;

public class MixtureOfHMMs {
	int MaxIter;
	int BG_maxIter;
	int BG_numState;
	int HMM_maxIter;
	int HMM_K;
	int HMM_M;
	int C; // The number of clusters (every cluster is corresponding to a hmm).
	ArrayList<HMM> hmms = new ArrayList<HMM>(C);
	double[][] seqsFracCounts;

	public MixtureOfHMMs(int MaxIter, int BG_numState, int BG_maxIter, int HMM_maxIter, int HMM_K, int HMM_M, int C) {
		this.MaxIter = MaxIter;
		this.BG_maxIter = BG_maxIter;
		this.BG_numState = BG_numState;
		this.HMM_maxIter = HMM_maxIter;
		this.HMM_K = HMM_K;
		this.HMM_M = HMM_M;
		this.C = C;
	}

	public void train(SequenceDataset data) {
		seqsFracCounts = new double[C][data.size()];
		initHMMs(data);
		for (int iter = 0; iter < MaxIter; iter++) {
			eStep(data);
			mStep(data);
		}
	}

	private void initHMMs(SequenceDataset data) {
		SplitDataByKMeans(data, true);
		for (int c = 0; c < C; ++c) {
			HMM hmm = new HMM(HMM_maxIter);
			hmm.train(data, HMM_K, HMM_M, seqsFracCounts[c]);
			hmms.add(hmm);
		}
	}

	private void SplitDataUniformly(SequenceDataset data) {
		for (int i = 0; i < data.size(); i++) {
			for (int c = 0; c < C; ++c) {
				seqsFracCounts[c][i] = 1.0 / c;
			}
		}
	}

	private void SplitDataRandomly(SequenceDataset data) {
		for (int i = 0; i < data.size(); i++) {
			double[] seqFracCounts = new double[C];
			for (int c = 0; c < C; ++c) {
				seqFracCounts[c] = new Random().nextDouble();
			}
			ArrayUtils.normalize(seqFracCounts);
			for (int c = 0; c < C; ++c) {
				seqsFracCounts[c][i] = seqFracCounts[c];
			}
		}
	}

	private void SplitDataByKMeans(SequenceDataset data, boolean useTwiceLongFeatures) {
		CheckinDataset bgd = new CheckinDataset();
		bgd.load(data);
		Background b = new Background(BG_maxIter);
		b.train(bgd, BG_numState);
		List<RealVector> featureVecs = new ArrayList<RealVector>(data.size());
		List<Double> weights = new ArrayList<Double>(data.size());
		for (int i = 0; i < data.size(); i++) {
			if (useTwiceLongFeatures) { // use BG_numState*2 dimension feature vectors
				RealVector featureVec = new ArrayRealVector(BG_numState * 2);
				for (int state = 0; state < BG_numState; ++state) {
					for (int n = 0; n < 2; ++n) {
						double membership = b.calcLL(data.getGeoDatum(2 * i + n), data.getTemporalDatum(2 * i + n),
								data.getTextDatum(2 * i + n));
						featureVec.setEntry(2 * state + n, membership);
					}
				}
				featureVecs.set(i, featureVec);
				weights.set(i, 1.0);
			} else { // use BG_numState dimension feature vectors
				RealVector featureVec = new ArrayRealVector(BG_numState);
				for (int state = 0; state < BG_numState; ++state) {
					double membership = b.calcLL(data.getGeoDatum(2 * i), data.getTemporalDatum(2 * i),
							data.getTextDatum(2 * i), data.getGeoDatum(2 * i + 1), data.getTemporalDatum(2 * i + 1),
							data.getTextDatum(2 * i + 1));
					featureVec.setEntry(state, membership);
				}
				featureVecs.set(i, featureVec);
				weights.set(i, 1.0);
			}
		}
		KMeans kMeans = new KMeans(500);
		List<Integer>[] kMeansResults = kMeans.cluster(featureVecs, weights, C);
		for (int c = 0; c < C; ++c) {
			List<Integer> cluster = kMeansResults[c];
			for (int i : cluster) {
				seqsFracCounts[c][i] = 1;
				for (int other_c = 0; other_c < C; ++other_c) {
					if (other_c != c) {
						seqsFracCounts[c][i] = 0;
					}
				}
			}
		}
	}

	private void eStep(SequenceDataset data) {
		for (int i = 0; i < data.size(); i++) {
			Sequence seq = data.getSequence(i);
			double[] posteriors = new double[C];
			for (int c = 0; c < C; ++c) {
				posteriors[c] = hmms.get(c).calcSeqScore(seq);
			}
			ArrayUtils.normalize(posteriors);
			for (int c = 0; c < C; ++c) {
				seqsFracCounts[c][i] = posteriors[c];
			}
		}
	}

	private void mStep(SequenceDataset data) {
		for (int c = 0; c < C; ++c) {
			HMM hmm = hmms.get(c);
			hmm.update(data, seqsFracCounts[c]);
		}
	}
}
