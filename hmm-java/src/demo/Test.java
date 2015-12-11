package demo;

import java.util.*;
import java.io.*;
import data.SequenceDataset;
import data.WordDataset;
import myutils.*;
import wordSimilarity.*;

public class Test {
	public static void main(String[] args) throws Exception {
		final String WorkPath = "/Users/keyangzhang/Documents/UIUC/Research/Mobility/mobility-master/";
		System.setOut(new PrintStream(new FileOutputStream(WorkPath + "results/result.txt")));
		// String datasetName = "ny40k.yaml";
		String datasetName = "4sq.yaml";
		String paraFile = args.length > 0 ? args[0] : WorkPath + "run/" + datasetName;

		Map config = new Config().load(paraFile);
		WordDataset wd = new WordDataset();
		SequenceDataset hmmd = new SequenceDataset();
		String wordFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("words");
		String sequenceFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("sequences");
		wd.load(wordFile);
		hmmd.load(sequenceFile);
		hmmd.setNumWords(wd.size());
		Map<Integer, String> dict = wd.getDict();

		WordSimilarity wordSimilarity = new WordSimilarity(hmmd);
		List<RankedObject> rankedWordPairs = new ArrayList<RankedObject>();
		int count = 0;
		for (Integer word1 : dict.keySet()) {
			++count;
			System.out.println(count);
			for (Integer word2 : dict.keySet()) {
				if (word1 > word2) {
					double similarityScore = wordSimilarity.getSimilarity(word1, word2);
					if (similarityScore > 0.5) {
						String gridNum1 = wordSimilarity.getStGridNum(word1).toString();
						String gridNum2 = wordSimilarity.getStGridNum(word2).toString();
						String ps = String.join("\t", dict.get(word1), dict.get(word2), gridNum1, gridNum2);
						rankedWordPairs.add(new RankedObject(ps, similarityScore));
					}
				}
			}
		}
		Collections.sort(rankedWordPairs);
		new CollectionFile<>(WorkPath + "results/wordSimilarity.txt").writeFrom(rankedWordPairs);
	}
}
