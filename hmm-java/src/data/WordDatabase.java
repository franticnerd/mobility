package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class WordDatabase implements Serializable {

  Map<Integer, String> dict = new HashMap<Integer, String>();

  // get a word by id
  public String getWord(int wordId) {
    return dict.get(wordId);
  }

  public int size() {
    return dict.size();
  }

  // load words from an input file, each line is: id + word
  public void load(String inputFile) throws Exception {
    BufferedReader br = new BufferedReader(new FileReader(inputFile));
    while (true) {
      String line = br.readLine();
      if (line == null)
        break;
      String[] items = line.split(",");
      int wordId = (new Integer(items[0])).intValue();
      dict.put(wordId, items[1]);
    }
    br.close();
  }

  public static void main(String [] args) throws Exception {
    String dataDir = "/Users/chao/Dataset/nyc_checkins/hmm/";
    String wordFile = dataDir + "words.txt";
    WordDatabase wd = new WordDatabase();
    wd.load(wordFile);
    System.out.println("Finished loading words. Count:" + wd.size());
  }
}
