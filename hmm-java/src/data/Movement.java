package data;

import java.util.ArrayList;
import java.util.List;

/**
 * This is the class for the length-2 movements and used in location prediction.
 * Created by chao on 5/2/15.
 */
public class Movement {

    int userId;
    List<Place> places = new ArrayList<Place>();

    public Movement(int userId, Place startPlace, Place endPlace) {
        this.userId = userId;
        places.add(startPlace);
        places.add(endPlace);
    }

    public int getUserId() {
        return userId;
    }

    public Place getStartPlace() {
        return places.get(0);
    }

    public Place getEndPlace() {
        return places.get(1);
    }

}
