package textAugmentation;

import java.util.*;
import myutils.*;
import data.Checkin;
import data.Sequence;
import data.SequenceDataset;

//compute and maintain word idf
public class WordIdf {
	HashMap<Integer, Double> word2idf = new HashMap<Integer, Double>();

	public WordIdf(SequenceDataset sequenceDataset) {
		for (Sequence sequence : sequenceDataset.getSequences()) {
			for (Checkin checkin : sequence.getCheckins()) {
				for (Integer word : checkin.getMessage().keySet()) {
					if (!word2idf.containsKey(word)) {
						word2idf.put(word, (double) 0);
					}
					word2idf.put(word, word2idf.get(word) + 1);
				}
			}
		}
		for (int word : word2idf.keySet()) {
			word2idf.put(word, 1 / Math.log(word2idf.get(word)));
		}
	}

	public Double getIdf(int word) {
		return word2idf.get(word);
	}
}
