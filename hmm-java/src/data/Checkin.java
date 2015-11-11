package data;

import java.io.Serializable;
import java.util.Map;

public class Checkin implements Serializable {

  int checkinId;
  int userId;
  int placeId;
  int timestamp;
  Map<Integer, Integer> message;  // Key: word, Value:count

  public Checkin(int checkinId, int userId, int placeId, int timestamp, Map<Integer, Integer> message) {
    this.checkinId = checkinId;
    this.userId = userId;
    this.placeId = placeId;
    this.timestamp = timestamp;
    this.message = message;
  }

  public int getId() {
    return checkinId;
  }

  public int getUserId() {
    return userId;
  }

  public int getPlaceId() {
    return placeId;
  }

  public int getTimestamp() {
    return timestamp;
  }

  public Map<Integer, Integer> getMessage() {
    return message;
  }


  // Get the text of the message and the description of the location
  public String getText() {
    String s = "";
    Place place = Database.getPlace(placeId);
    Map<Integer, Integer> description = place.getDescriptions();
    for (Map.Entry<Integer, Integer> e : description.entrySet()) {
      int wid = e.getKey();
      s += Database.getWord(wid) + " ";
    }
    for (Map.Entry<Integer, Integer> e : message.entrySet()) {
      int wid = e.getKey();
      s += Database.getWord(wid) + " ";
    }
    return s;
  }

}
