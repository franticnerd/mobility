package data;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Place implements Serializable {

  int placeId;
  Location loc;
  Map<Integer, Integer> description; // a list of word ids
  int popularity;  // the number of visitors.

  public Place(int placeId, Location loc, Map<Integer, Integer> description, int popularity) {
    this.placeId = placeId;
    this.loc = loc;
    this.description = description;
    this.popularity = popularity;
  }

  public int getId() {
    return placeId;
  }

  public Location getLocation() {
    return loc;
  }

  public Map<Integer, Integer> getDescriptions() {
    return description;
  }

  public int getPopularity() {
    return popularity;
  }

  public void addMessage(Map<Integer, Integer> message) {
    Iterator iter = message.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<Integer, Integer> entry = (Map.Entry<Integer, Integer>) iter.next();
      int wordId = entry.getKey();
      int count = entry.getValue();
      if (description.containsKey(wordId))
        continue;
//        description.put(wordId, description.get(wordId)+count);
      else
        description.put(wordId, count);
    }
  }

  public double getEuclideanDist(Place p) {
    return loc.calcEuclideanDist(p.getLocation());
  }

  public double getGeographicDist(Place p) {
    return loc.calcGeographicDist(p.getLocation());
  }

  public Place copy() {
    Map<Integer, Integer> newDescription = new HashMap<Integer, Integer>(description);
    return new Place(placeId, loc, newDescription, popularity);
  }

}
