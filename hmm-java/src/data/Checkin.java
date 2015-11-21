package data;

import java.io.Serializable;
import java.util.Map;

public class Checkin implements Serializable {

  int checkinId;
  long userId;
  Location location;
  int timestamp;
  Map<Integer, Integer> message;  // Key: word, Value:count

  public Checkin(int checkInId, int timestamp, long userId, double lat, double lng, Map<Integer, Integer> message) {
    this.checkinId = checkInId;
    this.timestamp = timestamp;
    this.userId = userId;
    this.location = new Location(lat, lng);
    this.timestamp = timestamp;
    this.message = message;
  }

  public int getId() {
    return checkinId;
  }

  public long getUserId() {
    return userId;
  }

  public Location getLocation() {
    return location;
  }

  public int getTimestamp() {
    return timestamp;
  }

  public Map<Integer, Integer> getMessage() {
    return message;
  }

  // Get the text of the message and the description of the location
  public String getText(WordDataset wd) {
    String s = "";
    for (Map.Entry<Integer, Integer> e : message.entrySet()) {
      int wid = e.getKey();
      s += wd.getWord(wid) + " ";
    }
    return s;
  }

}
