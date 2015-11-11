package outlier;

import data.Movement;
import data.MovementDB;
import myutils.ScoreCell;
import myutils.TopKSearcher;

/**
 * Created by chao on 5/4/15.
 */
public abstract class OutlierDetecter {

    int [] outliers;
    
    // retrieve a set of outliers from mdb.
    public void detect(MovementDB mdb, int K) {
        TopKSearcher tks = new TopKSearcher();
        tks.init(K);
        for (int i=0; i<mdb.size(); i++) {
            Movement m = mdb.getMovement(i);
            ScoreCell sc = calcScore(mdb, m, i);
            tks.add(sc);
            System.out.println(sc.getScore());
        }
        ScoreCell [] topKResults = new ScoreCell[K];
        topKResults = tks.getTopKList(topKResults);
        outliers = new int[K];
        for (int i=0; i<K; i++) {
            outliers[i] = topKResults[i].getId();
        }
    }

    public abstract ScoreCell calcScore(MovementDB mdb, Movement m, int movementId);

    public void printOutliers() {
        for (int i=0; i<outliers.length; i++) {
            System.out.print(outliers[i] + " ");
        }
        System.out.print("\n");
    }
}
