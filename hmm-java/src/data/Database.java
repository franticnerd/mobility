package data;


/**
 * Created by chao on 4/29/15.
 */
public class Database {

    public static PlaceDatabase pd; // The place database.
    public static WordDatabase wd; // The word database.
    public static CheckinDatabase cd; // The checkin database.
    public static SequenceDatabase sd; // The checkin database.

    public static void loadData(String placeFile, String checkinFile, String wordFile) throws Exception {
        pd = new PlaceDatabase();
        pd.load(placeFile);
        System.out.println("Finished loading places. Count:" + pd.size());
        cd = new CheckinDatabase();
        cd.load(checkinFile);
        System.out.println("Finished loading checkins. Count:" + cd.size());
        wd = new WordDatabase();
        wd.load(wordFile);
        System.out.println("Finished loading words. Count:" + wd.size());
        sd = new SequenceDatabase();
        sd.loadCheckins(cd);
        System.out.println("Finished loading sequences. Count:" + sd.size());
    }

    public static int numPlace() {
        return pd.size();
    }

    public static int numWord() {
        return wd.size();
    }

    public static Place getPlace(int placeId) {
        return pd.getPlace(placeId);
    }

    public static Checkin getCheckin(int checkinId) {
        return cd.getCheckin(checkinId);
    }

    public static String getWord(int wordId) {
        return wd.getWord(wordId);
    }

    public static Sequence getSequence(int userId) {
        return sd.getSequence(userId);
    }

}
