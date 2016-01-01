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

public class MixtureOfHMMs_User {
	int MaxIter;
	int BG_maxIter;
	int BG_numState;
	int HMM_maxIter;
	int HMM_K;
	int HMM_M;
	int C; // The number of clusters (every cluster is corresponding to a hmm).
	ArrayList<HMM> hmms = new ArrayList<HMM>(C);
	double[][] seqsFracCounts;
	HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();

	public MixtureOfHMMs_User(int MaxIter, int BG_numState, int BG_maxIter, int HMM_maxIter, int HMM_K, int HMM_M,
			int C) {
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
		calcUser2seqs(data);
		initHMMs(data);
		for (int iter = 0; iter < MaxIter; iter++) {
			eStep(data);
			mStep(data);
		}
	}

	public void calcUser2seqs(SequenceDataset data) {
		for (int i = 0; i < data.size(); i++) {
			Sequence seq = data.getSequence(i);
			long user = seq.getUserId();
			if (!user2seqs.containsKey(user)) {
				user2seqs.put(user, new HashSet<Integer>());
			}
			user2seqs.get(user).add(i);
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
		for (long user : user2seqs.keySet()) {
			double[] seqFracCounts = new double[C];
			for (int c = 0; c < C; ++c) {
				seqFracCounts[c] = new Random().nextDouble();
			}
			ArrayUtils.normalize(seqFracCounts);
			for (int i : user2seqs.get(user)) {
				for (int c = 0; c < C; ++c) {
					seqsFracCounts[c][i] = seqFracCounts[c];
				}
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
		HashMap<Integer, Long> u2user = new HashMap<Integer, Long>(); // u is the index of user
		int u = 0;
		for (long user : user2seqs.keySet()) {
			RealVector featureVec;
			if (useTwiceLongFeatures) { // use BG_numState*2 dimension feature vectors
				featureVec = new ArrayRealVector(BG_numState * 2);
				for (int state = 0; state < BG_numState; ++state) {
					for (int n = 0; n < 2; ++n) {
						for (int i : user2seqs.get(user)) {
							double membership = b.calcLL(data.getGeoDatum(2 * i + n), data.getTemporalDatum(2 * i + n),
									data.getTextDatum(2 * i + n));
							featureVec.addToEntry(2 * state + n, membership);
						}
						featureVec.setEntry(2 * state + n, Math.exp(featureVec.getEntry(2 * state + n))); // transform to probability
					}
				}
			} else { // use BG_numState dimension feature vectors
				featureVec = new ArrayRealVector(BG_numState);
				for (int state = 0; state < BG_numState; ++state) {
					for (int i : user2seqs.get(user)) {
						double membership = b.calcLL(data.getGeoDatum(2 * i), data.getTemporalDatum(2 * i),
								data.getTextDatum(2 * i), data.getGeoDatum(2 * i + 1), data.getTemporalDatum(2 * i + 1),
								data.getTextDatum(2 * i + 1));
						featureVec.addToEntry(state, membership);
					}
					featureVec.setEntry(state, Math.exp(featureVec.getEntry(state))); // transform to probability
				}
			}
			featureVecs.set(u, featureVec);
			weights.set(u, 1.0);
			u2user.put(u, user);
			++u;
		}
		KMeans kMeans = new KMeans(500);
		List<Integer>[] kMeansResults = kMeans.cluster(featureVecs, weights, C);
		for (int c = 0; c < C; ++c) {
			List<Integer> members = kMeansResults[c];
			for (int member : members) {
				long user = u2user.get(member);
				for (int i : user2seqs.get(user)) {
					seqsFracCounts[c][i] = 1;
					for (int other_c = 0; other_c < C; ++other_c) {
						if (other_c != c) {
							seqsFracCounts[c][i] = 0;
						}
					}
				}
			}
		}
	}

	private void eStep(SequenceDataset data) {
		for (long user : user2seqs.keySet()) {
			double[] posteriors = new double[C];
			for (int c = 0; c < C; ++c) {
				for (int i : user2seqs.get(user)) {
					Sequence seq = data.getSequence(i);
					posteriors[c] += hmms.get(c).calcSeqScore(seq);
				}
				posteriors[c] = Math.exp(posteriors[c]);
			}
			ArrayUtils.normalize(posteriors);
			for (int i : user2seqs.get(user)) {
				for (int c = 0; c < C; ++c) {
					seqsFracCounts[c][i] = posteriors[c];
				}
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
