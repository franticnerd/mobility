package textAugmentation;

import data.SequenceDataset;
import data.WordDataset;

import java.util.*;
import java.io.*;
import myutils.*;
import distribution.*;

//should instantiate only once as it will compute "WordIdf" and "WordSimilarity", which is time-consuming
public class Augmenter {
	WordDataset wd = null;
	WordIdf wordIdf = null;
	HashMap<Integer, HashMap<Integer, Double>> similarities = null;

	public Augmenter(SequenceDataset sequenceDataset, WordDataset wd) {
		this.wd = wd;
		wordIdf = new WordIdf(sequenceDataset);
		WordSimilarity ws = new WordSimilarity(sequenceDataset, wd);
		similarities = ws.getSimilarities();
	}

	public Augmenter(SequenceDataset sequenceDataset, WordDataset wd, String filePath) throws Exception {
		this.wd = wd;
		wordIdf = new WordIdf(sequenceDataset);
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
		WordSimilarity ws = (WordSimilarity) ois.readObject();
		similarities = ws.getSimilarities();
	}

	public Map<Integer, Integer> getAugmentedText(Map<Integer, Integer> text, int augmentedSize) {
		Map<Integer, Integer> augmentedText = new HashMap<Integer, Integer>(text);
		HashMap<Integer, Double> word2idf = new HashMap<Integer, Double>();
		for (int word : text.keySet()) {
			word2idf.put(word, wordIdf.getIdf(word));
			augmentedSize -= text.get(word);
		}
		Categorical c1 = new Categorical(word2idf);
		for (int i = 0; i < augmentedSize; ++i) {
			int word = (Integer) c1.sample();
			if (similarities.containsKey(word)) {
				Categorical c2 = new Categorical(similarities.get(word));
				int addedWord = (Integer) c2.sample();
				if (!augmentedText.containsKey(addedWord)) {
					augmentedText.put(addedWord, 0);
				}
				augmentedText.put(addedWord, augmentedText.get(addedWord) + 1);
			}
		}
		return augmentedText;
	}
}
