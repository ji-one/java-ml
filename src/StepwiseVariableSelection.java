import java.util.Arrays;
import java.util.Comparator;
import static org.junit.Assert.*;

public abstract class StepwiseVariableSelection implements VariableSelection{
	public Strategy strategy;
	public DataSet dataSet;
	public boolean[] isEliminatedAttr;
	public int numSets = 8;
	public double[] instanceWeights;
	public int kOpt = 7;
	public DataSet getDataSet() {
		return dataSet;
	}

	public double[] getInstanceWeights() {
		return instanceWeights;
	}

	public Strategy getStrategy() {
		return strategy;
	}

	public void setIsEliminatedAttr(boolean[] isEliminatedAttr) {
		this.isEliminatedAttr = isEliminatedAttr;
	}
	public void setkOpt(int kOpt) {
		this.kOpt = kOpt;
	}

	public void setInstanceWeights(double[] instanceWeights) {
		this.instanceWeights = instanceWeights;
	}
	protected double calcError(int[][] kNNindices) {
		double error = 0.0;
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			boolean isWrongPredict = voteCount(kNNindices[i]) != this.dataSet.trainLabel[i];
			if (isWrongPredict)
				error++;
		}
		return error;
	}
	
	protected abstract int voteCount(int[] is);
	protected int[][] orderedIndices(double[][] distances) {
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] alpha = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				alpha[j] = j;
			}
			exComparator comparator = new exComparator(distances[i], i);
			comparator.descending = true;
			Arrays.sort(alpha, comparator);
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = alpha[j];
			}
		}
		return orderedIndices;
	}
	
	protected abstract double getDistance(int[] vector1, int[] vector2);
	protected int[][] computeNewNearestK(int[][] orderedIndice, double[][] temporaryDistances) {
		// compute new k nearest
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			// annoying Integer to int casting issues
			Integer[] integers = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndice.length; j++) {
				integers[j] = orderedIndice[i][j];
			}
			exComparator comparator = new exComparator(temporaryDistances[i], i);
			comparator.descending = true;
			Arrays.sort(integers, comparator);
			for (int j = 0; j < orderedIndice.length; j++) {
				orderedIndices[i][j] = integers[j];
			}
		}
		return orderedIndices;
	}
	protected double[][] linearDistanceUpdate(double[][] distances, int m) {
		// linear-time distance update
		double[][] temporaryDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				double distanceCalculation = strategy.getDistanceStrategy().calcDistance(this.dataSet.trainEx[i][m], this.dataSet.trainEx[j][m]);
				temporaryDistances[i][j] = distances[i][j] - distanceCalculation;
				temporaryDistances[j][i] = temporaryDistances[i][j];
			}
		}
		return temporaryDistances;
	}
	protected double[][] setCrossValidationDistance() {
		// calculate all distances to avoid recomputation
		double[][] distances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;

			for (int t = from; t < to; t++) {
				for (int s = t+1; s < to; s++) {
					distances[t][s] = Double.POSITIVE_INFINITY;
					distances[s][t] = distances[t][s];
				}
			}
		}
		return distances;
	}
	
	public class exComparator implements Comparator {
		public boolean descending;
		private double[] dists;
	    private int[] ex;
	    private int exIndex = -1;
	    public boolean isDescending = false;
	    
	    exComparator(int[] ex) {
	      this.ex = ex;
	    }
	    
	    /* Constructor which assumes base ex may be used in comparison */
	    public exComparator(int[] ex, int exIndex) {
		  this.ex = ex;
		  this.exIndex = exIndex;
		}
	    
	    /* Constructor which allows for precomputed distances */
	    protected exComparator(double[] dists, int exIndex) {
			this.dists = dists;
			this.exIndex = exIndex;
		}

	    public int compare(Object object1, Object object2) {
	    	int trainExIndex1 = (int)(Integer)object1;
	    	int trainExIndex2 = (int)(Integer)object2;
	    	
	    	// ignore if training example is in data set
	    	if (trainExIndex1 == this.exIndex) return 1;
	    	if (trainExIndex2 == this.exIndex) return -1;
	        
	        // take column of min distance
	    	double dist1;
	    	double dist2;
	    	if (this.dists == null) {
	    	    dist1 = getDistance(dataSet.trainEx[trainExIndex1], this.ex);
	    		dist2 = getDistance(dataSet.trainEx[trainExIndex2], this.ex);
	    	}
	    	else {
	    		dist1 = this.dists[trainExIndex1];
	    		dist2 = this.dists[trainExIndex2];	    		
	    	}
	    	int result = 0;
	    	if (dist1 > dist2) result = -1;
    		if (dist2 > dist1) result = 1;
    		if (this.isDescending) result *= -1;
    		return result;
	    }
	}

	
}
