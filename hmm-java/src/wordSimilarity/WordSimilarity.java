package wordSimilarity;

import java.util.*;
import myutils.*;
import data.Checkin;
import data.Sequence;
import data.SequenceDataset;

public class WordSimilarity {
	final int LngGridNum = 10;
	final int LatGridNum = 10;
	HashMap<Integer,HashMap<StGrid,Double>> word2stGrids = new HashMap<Integer,HashMap<StGrid,Double>>();
	
	public WordSimilarity(SequenceDataset sequenceDataset) {
		double lngMax = Double.MIN_VALUE;
		double latMax = Double.MIN_VALUE;
		double lngMin = Double.MAX_VALUE;
		double latMin = Double.MAX_VALUE;
		for(Sequence sequence:sequenceDataset.getSequences()){
			for(Checkin checkin:sequence.getCheckins()){
				double lng = checkin.getLocation().getLng();
				double lat = checkin.getLocation().getLat();
				if(lng>lngMax){
					lngMax = lng;
				}
				if(lat>latMax){
					latMax = lat;
				}
				if(lng<lngMin){
					lngMin = lng;
				}
				if(lat<latMin){
					latMin = lat;
				}
			}
		}
		for(Sequence sequence:sequenceDataset.getSequences()){
			for(Checkin checkin:sequence.getCheckins()){
				Calendar calendar = Calendar.getInstance();
				calendar.setTimeInMillis(checkin.getTimestamp()*1000);
				int timeGrid = calendar.get(Calendar.HOUR_OF_DAY);
				double lng = checkin.getLocation().getLng();
				double lat = checkin.getLocation().getLat();
				int lngGrid = (int) Math.ceil( (lng-lngMin)*LngGridNum / (lngMax-lngMin) );
				int latGrid = (int) Math.ceil( (lat-latMin)*LatGridNum / (latMax-latMin) );
				StGrid stGrid = new StGrid(lngGrid, latGrid, timeGrid);
//				System.out.print(stGrid);
				for(Integer word:checkin.getMessage().keySet()){
					if(!word2stGrids.containsKey(word)){
						word2stGrids.put(word, new HashMap<StGrid,Double>());
					}
					HashMap<StGrid,Double> stGrids = word2stGrids.get(word);
					if(!stGrids.containsKey(stGrid)){
						stGrids.put(stGrid, (double) 0);
					}
					stGrids.put(stGrid, stGrids.get(stGrid)+1);
				}
			}
		}
	}
	
	public Integer getStGridNum(int word){
		return word2stGrids.get(word).size();
	}
	
	public double getSimilarity(int word1, int word2){
		HashMap<StGrid,Double> stGrids1 = word2stGrids.get(word1);
		HashMap<StGrid,Double> stGrids2 = word2stGrids.get(word2);
		return new MapVectorUtils<StGrid>().cosine(stGrids1, stGrids2);
	}
}
