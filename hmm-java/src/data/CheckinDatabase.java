package data;

/* *
 * This class represents a database of checkins.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class CheckinDatabase implements Serializable {

  Map<Integer, Checkin> checkins = new HashMap<Integer, Checkin>();

  // get a checkin by id
  public Checkin getCheckin(int checkinId) {
    return checkins.get(checkinId);
  }


  public int size() {
    return checkins.size();
  }


  public List<Checkin> getAllCheckins() {
    List<Checkin> checkinList = new ArrayList<Checkin>();
    Iterator iter = checkins.entrySet().iterator();
    while(iter.hasNext()) {
      Map.Entry<Integer, Checkin> entry = (Map.Entry<Integer, Checkin>) iter.next();
      checkinList.add(entry.getValue());
    }
    return checkinList;
  }

  // load checkins from an input file, each line is a checkin.
  public void load(String inputFile) throws IOException {
    BufferedReader br = new BufferedReader(new FileReader(inputFile));
    while(true) {
      String line = br.readLine();
      if(line == null) break;
      Checkin checkin = parseCheckin(line);
      checkins.put(checkin.getId(), checkin);
    }
    br.close();
  }

  // Each line contains: checkin Id, userId, placeid, timestamp, message
  private Checkin parseCheckin(String line) {
    String[] items = line.split(",");
    int checkinId = new Integer(items[0]).intValue();
    int userId = new Integer(items[1]).intValue();
    int placeId = new Integer(items[2]).intValue();
    int timestamp = new Integer(items[3]).intValue();
    Map<Integer, Integer> message = null;
    if(items.length < 5)
      message = new HashMap<Integer, Integer>();
    else
      message = parseMessage(items[4]);
    return new Checkin(checkinId, userId, placeId, timestamp, message);
  }

  private Map<Integer, Integer> parseMessage(String s) {
    Map<Integer, Integer> message = new HashMap<Integer, Integer>();
    String [] items = s.split("\\s");
    for (int i = 0; i < items.length; i++) {
      int wordId = new Integer(items[i]).intValue();
      if(message.containsKey(wordId)) {
//        message.put(wordId, message.get(wordId) + 1);
        continue;
      } else {
        message.put(wordId, 1);
      }
    }
    return message;
  }


  public static void main(String [] args) throws Exception {
    String dataDir = "/Users/chao/Dataset/nyc_checkins/hmm/";
    String checkinFile = dataDir + "checkins.txt";
    CheckinDatabase cd = new CheckinDatabase();
    cd.load(checkinFile);
    System.out.println("Finished loading checkins. Count:" + cd.size());
  }

}
