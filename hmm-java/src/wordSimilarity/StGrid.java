package wordSimilarity;

// "StGrid" stands for "spatial-temporal grid"
public class StGrid {
	int lngGrid;
	int latGrid;
	int timeGrid;
	
	public StGrid(int lngGrid, int latGrid, int timeGrid) {
		this.lngGrid = lngGrid;
		this.latGrid = latGrid;
		this.timeGrid = timeGrid;
	}
	
	@Override
	public String toString() {
		return lngGrid+"\t"+latGrid+"\t"+timeGrid+"\n";
	}
}
