package data;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * A database that consists of multiple movemnts.
 * Also, for each movement, it has a number of candidate places that are used for prediction.
 * Created by chao on 5/2/15.
 */
public class MovementDB {

    List<Movement> movements = new ArrayList<Movement>();
    // For each movement, we keep a set of candidate end places.
    List<List<Place>> candidates = new ArrayList<List<Place>>();

    public int size() {
        return movements.size();
    }

    public List<Movement> getMovements() {
        return movements;
    }

    public Movement getMovement(int i) {
        return movements.get(i);
    }

    public List<Place> getCandidate(int i) {
        return candidates.get(i);
    }

    // load places from an input file, each line is a sequence.
    public void load(String inputFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        while(true) {
            String line = br.readLine();
            if(line == null) break;
            if(line.isEmpty()) continue;
            String[] items = line.split(",");
            int cid1 = new Integer(items[0]).intValue();
            int cid2 = new Integer(items[1]).intValue();
            Checkin c1 = Database.getCheckin(cid1);
            Checkin c2 = Database.getCheckin(cid2);
            Place startPlace = Database.getPlace(c1.getPlaceId()).copy();
//            startPlace.addMessage(c1.getMessage());
            Place endPlace = Database.getPlace(c2.getPlaceId());
            int userId = new Integer(items[2]).intValue();
            Movement m = new Movement(userId, startPlace, endPlace);
            movements.add(m);
        }
        br.close();
    }


//    // Get the candidate destinations for the first place.
//    public void genCandidates(double dist) {
//        for (Movement m : movements) {
//            Place startPlace = m.getStartPlace();
//            List<Integer> neighborIds = Database.pd.getNeighbors(startPlace.getId(), dist);
//            List<Place> neighbors = new ArrayList<Place>();
//            for (Integer id : neighborIds)
//                neighbors.add(Database.pd.getPlace(id));
//            neighbors.add(m.getEndPlace());  // Ensure the ground truth place is also in the list.
//            System.out.println("Candidate size:" + neighbors.size());
//            candidates.add(neighbors);
//        }
//    }

    // Get the candidate destinations for the first place.
    public void genCandidates() {
        for (Movement m : movements) {
            int userId = m.getUserId();
            List<Checkin> checkins = Database.getSequence(userId).getCheckins();
            List<Place> neighbors = new ArrayList<Place>();
            for (Checkin c : checkins)
                neighbors.add(Database.pd.getPlace(c.getPlaceId()));
//            System.out.println("Candidate size:" + neighbors.size());
            candidates.add(neighbors);
        }
    }

}
