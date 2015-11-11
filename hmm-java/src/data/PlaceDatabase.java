package data;

/* *
 * This class represents a database of places.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class PlaceDatabase implements Serializable {

  Map<Integer, Place> places = new HashMap<Integer, Place>();

  public PlaceDatabase() {}

  public PlaceDatabase(Map<Integer, Place> places) {
    this.places = places;
  }

  // get a place by id
  public Place getPlace(int placeId) {
    return places.get(placeId);
  }

  public int size() {
    return places.size();
  }

  public Map<Integer, Place> getPlaces() {
    return places;
  }

  // load places from an input file, each line is a place.
  public void load(String inputFile) throws IOException {
    BufferedReader br = new BufferedReader(new FileReader(inputFile));
    while(true) {
      String line = br.readLine();
      if(line == null) break;
      if(line.isEmpty()) continue;
      Place place = parsePlace(line);
      places.put(place.getId(), place);
    }
    br.close();
  }

  // Each line contains: place Id, userId, placeid, timestamp, message
  private Place parsePlace(String line) {
    String[] items = line.split(",");
    int placeId = new Integer(items[0]).intValue();
    double lng = new Double(items[1]).doubleValue();
    double lat = new Double(items[2]).doubleValue();
    Location location = new Location(lng, lat);
    Map<Integer, Integer> description = parseDescription(items[3]);
    int popularity = new Integer(items[4]).intValue();
    return new Place(placeId, location, description, popularity);
  }


  private Map<Integer, Integer> parseDescription(String s) {
    Map<Integer, Integer> description = new HashMap<Integer, Integer>();
    if(s.isEmpty())
      return description;
    String [] items = s.split("\\s+");
    for (int i = 0; i < items.length; i++) {
      int wordId = new Integer(items[i]).intValue();
      if(description.containsKey(wordId)) {
//        description.put(wordId, description.get(wordId) + 1);
        continue;
      } else {
        description.put(wordId, 1);
      }
    }
    return description;
  }

  public PlaceDatabase copy() {
    Map<Integer, Place> newPlaces = new HashMap<Integer, Place> ();
    Iterator iter = places.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<Integer, Place> entry = (Map.Entry<Integer, Place>) iter.next();
      int placeId = entry.getKey();
      Place place = entry.getValue();
      newPlaces.put(placeId, place.copy());
    }
    return new PlaceDatabase(newPlaces);
  }

  // Get the list of places that are close to the given place.
  public List<Integer> getNeighbors(int queryPlaceId, double dist) {
    Place queryPlace = places.get(queryPlaceId);
    List<Integer> results = new ArrayList<Integer>();
    Iterator iter = places.entrySet().iterator();
    while(iter.hasNext()) {
      Map.Entry entry = (Map.Entry) iter.next();
      int placeId = (Integer)entry.getKey();
      Place place = (Place) entry.getValue();
      if(placeId != queryPlaceId && place.getGeographicDist(queryPlace) <= dist)
        results.add(placeId);
    }
    return results;
  }


  public static void main(String [] args) throws Exception {
    String dataDir = "/Users/chao/Dataset/nyc_checkins/hmm/";
    String placeFile = dataDir + "venues.txt";
    PlaceDatabase pd = new PlaceDatabase();
    pd.load(placeFile);
    System.out.println("Finished loading places. Count:" + pd.size());
  }

}
