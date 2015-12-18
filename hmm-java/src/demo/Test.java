package demo;

import java.util.*;
import java.io.*;

import data.Checkin;
import data.Sequence;
import data.SequenceDataset;
import data.WordDataset;
import myutils.*;
import textAugmentation.*;

//entrance function used by Keyang for testing
public class Test {
	static public final String WorkPath = "/Users/keyangzhang/Documents/UIUC/Research/Mobility/mobility/";

	public static void main(String[] args) throws Exception {
		System.setOut(new PrintStream(new FileOutputStream(WorkPath + "results/result.txt")));
//		 String datasetName = "ny40k.yaml";
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

		WordSimilarity ws = new WordSimilarity(hmmd, wd);
//		ws.printHighlySimilarPairs();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(WorkPath + "results/similarity.dat"));
		oos.writeObject(ws);
		oos.flush();
		
		Augmentation augmentation = new Augmentation(hmmd, wd, WorkPath + "results/similarity.dat");
		for (Sequence sequence : hmmd.getSequences()) {
			for (Checkin checkin : sequence.getCheckins()) {
				Map<Integer, Integer> text = checkin.getMessage();
				Map<Integer, Integer> augmentedText = augmentation.getAugmentedText(text,30);
//				checkin.setMessage(augmentedText);
				for(int word:text.keySet()){
					System.out.print(wd.getWord(word)+"="+text.get(word)+", ");
				}
				System.out.println();
				for(int word:augmentedText.keySet()){
					System.out.print(wd.getWord(word)+"="+augmentedText.get(word)+", ");
				}
				System.out.println("\n");
			}
		}
	}
}
