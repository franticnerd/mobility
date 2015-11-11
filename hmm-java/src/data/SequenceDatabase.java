package data;

/* *
 * This class represents the checkin sequences of multiple users.
 */

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.Serializable;
import java.util.*;

public class SequenceDatabase implements Serializable {

  Map<Integer, Sequence> sequences = new HashMap<Integer, Sequence>();

  public Sequence getSequence(int userId) {
    return sequences.get(userId);
  }

  // load checkins to update sequences.
  public void loadCheckins(CheckinDatabase cd) {
    List<Checkin> checkins = cd.getAllCheckins();
    for(Checkin checkin : checkins) {
      int userId = checkin.getUserId();
      if(sequences.containsKey(userId)) {
        sequences.get(userId).addCheckin(checkin);
      } else {
        Sequence sequence = new Sequence(userId);
        sequence.addCheckin(checkin);
        sequences.put(userId, sequence);
      }
    }
    sort();
  }

  public void sort() {
    Iterator iter = sequences.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<Integer, Sequence> entry = (Map.Entry<Integer, Sequence>) iter.next();
      Sequence userSequence = entry.getValue();
      userSequence.sortCheckins();
    }
  }

  public int size() {
    return sequences.size();
  }

  // Extract the dense length-2 sequences
  public List<Sequence> getDenseSequences(double minGap, double maxGap) {
    List<Sequence> result = new ArrayList<Sequence>();
    Iterator iter = sequences.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<Integer, Sequence> entry = (Map.Entry<Integer, Sequence>) iter.next();
      Sequence userSequence = entry.getValue();
      if (userSequence.size() < 2)
        continue;
      List<Sequence> oneUserDenseSequences = getDenseSequencesForOneUser(userSequence, minGap, maxGap);
      result.addAll(oneUserDenseSequences);
    }
    return result;
  }


  private List<Sequence> getDenseSequencesForOneUser(Sequence sequence, double minGap, double maxGap) {
    List<Sequence> result = new ArrayList<Sequence>();
    for (int i=0; i<sequence.size()-1; i++) {
      Checkin c1 = sequence.getCheckin(i);
      Checkin c2 = sequence.getCheckin(i+1);
      if (c2.getTimestamp() - c1.getTimestamp() <= maxGap &&
          c2.getTimestamp() - c1.getTimestamp() >= minGap &&
              c1.getPlaceId() != c2.getPlaceId()) {
        Sequence s = new Sequence(c1.getUserId());
        s.addCheckin(c1);
        s.addCheckin(c2);
        result.add(s);
      }
    }
    return result;
  }

  public void write(List<Sequence> seqs, String outputFile) throws Exception {
    BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
    for (Sequence seq: seqs)
      bw.append(seq.toString());
    bw.close();
  }

}
