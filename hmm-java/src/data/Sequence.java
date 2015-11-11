package data;

/* *
 * This class represents a checkin sequence of a user.
 */

import java.io.Serializable;
import java.util.*;

public class Sequence implements Serializable {

  int userId;

  // a list of checkins for the user
  List<Checkin> checkins = null;

  public Sequence() {}

  public Sequence(int userId) {
    this.userId = userId;
    checkins = new ArrayList<Checkin>();
  }

  public Sequence(int userId, List<Checkin> checkins) {
    this.userId = userId;
    this.checkins = checkins;
  }

  public List<Checkin> getCheckins() {
    return checkins;
  }

  public Checkin getCheckin(int index) {
    return checkins.get(index);
  }

  public int size() {
    return checkins.size();
  }

  public void addCheckin(Checkin c) {
    checkins.add(c);
  }

  public void sortCheckins() {
    Collections.sort(checkins, new Comparator<Checkin>() {
      public int compare(Checkin c1, Checkin c2) {
        if (c1.getTimestamp() - c2.getTimestamp() > 0)
          return 1;
        else if (c1.getTimestamp() - c2.getTimestamp() == 0)
          return 0;
        else
          return -1;
      }
    });
  }


  // convert a length-2 sequence into a string
  public String toStringForLengthTwo() {
    Checkin c1 = checkins.get(0);
    Checkin c2 = checkins.get(1);
    String s = c1.getId() + "," + c2.getId() + ",";
    s += c1.getUserId() + ",";
    Place p1 = Database.getPlace(c1.getPlaceId());
    Place p2 = Database.getPlace(c2.getPlaceId());
    s += p1.getGeographicDist(p2);
    s += ",";
    s += c2.getTimestamp() - c1.getTimestamp();
    s += "," + c1.getText() + "," + c2.getText();
    return s;
  }

}
