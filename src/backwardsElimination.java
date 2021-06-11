import java.util.Arrays;
import java.util.Comparator;

public class backwardsElimination extends StepwiseVariableSelection {

	public void variableSelection(DataSet dataSet, Strategy strategy, boolean[] isEliminatedAttr,
			double[] instanceWeights) {
		backwardsEliminations(dataSet, strategy, isEliminatedAttr, instanceWeights);
		double[][] distances = distanceSettingForBackward();
		
		int[][] orderedIndices = orderedIndices(distances);
		
		// calculate base error with no attribute elimination
		double baselineError = calcError(orderedIndices);
		
		distances = iterateEachAttribute(distances, orderedIndices, baselineError);
		for (int i = 0; i < distances.length; i++) {
			double[] inArr = distances[i];
			for (int j = 0; j < distances.length; j++) {
			System.out.print(inArr[j] + " ");
			}
			System.out.println();
			}

		
	}
	private void backwardsEliminations(DataSet dataSet, Strategy strategy, boolean[] isEliminatedAttr,
			double[] instanceWeights) {
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;
	}
	public backwardsElimination() {
		
	}
	public backwardsElimination(DataSet dataSet, Strategy strategy) {
		this.dataSet = dataSet;
		this.strategy = strategy;
	}
	public backwardsElimination(DataSet dataSet, Strategy strategy, boolean[] isEliminatedAttr,
			double[] instanceWeights) {
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;
	}
	protected double[][] iterateEachAttribute(double[][] distances, int[][] orderedIndices, double baselineError) {
		int sum = 0;
		double[][] newdistance = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		int[][] orderedIndice = orderedIndices;
		double baselineerror = baselineError;
		// iterate over each attribute
		for (int m = 0; m < this.dataSet.numAttrs; m++) {
			double[][] temporaryDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
			temporaryDistances = linearDistanceUpdate(distances, m);
			orderedIndice = computeNewNearestK(orderedIndice, temporaryDistances);
			double adjustedError = calcError(orderedIndice);

			// if error improved, keep attribute eliminated; else, retain
			boolean errorImproved = adjustedError < baselineerror;
			if ( errorImproved ) {
				this.isEliminatedAttr[m] = true;
				baselineerror = adjustedError;
				newdistance = temporaryDistances;
				sum++;
			}
			//if(m == this.dataSet.numAttrs - 1)
			//	System.out.printf("%d attributes removed.\n", sum);
		}
		
		
		
		return newdistance;
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
	protected double[][] setCrossValidationDistancefortest() {
		// calculate all distances to avoid recomputation
		this.numSets = 1;
		this.dataSet.numTrainExs = 2;
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
	protected double[][] distanceSettingForBackward() {
		double[][] distances = setCrossValidationDistance();
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				if (distances[i][j] == Double.POSITIVE_INFINITY) continue;
				
				distances[i][j] = getDistance(this.dataSet.trainEx[i], this.dataSet.trainEx[j]);
				distances[j][i] = distances[i][j];
			}
		}
		return distances;
	}
	protected double getDistance(int[] vector1, int[] vector2) {
		int len = Math.min(vector1.length, vector2.length);
		int distance = 0;
		for (int i = 0; i < len; i++) {
			// skip if attribute is eliminated
			if (this.isEliminatedAttr[i] == true) continue;
			distance += this.strategy.getDistanceStrategy().calcDistance(vector1[i], vector2[i]);
		}
        return distance;
    }
	protected int voteCount(int[] indices) {
		double vote_1 = 0;
    	double vote_0 = 0;
    	int len = Math.min(indices.length, this.kOpt);
    	for (int k = 0; k < len; k++) {
    		int index = indices[k];
    		if (this.dataSet.trainLabel[index] == 1)
    			vote_1 += this.instanceWeights[index];
    		else
    			vote_0 += this.instanceWeights[index];
    	}
    	
    	return (vote_1 > vote_0)? 1 : 0;
	}
	public boolean[] getIsEliminatedAttr() {
		return isEliminatedAttr;
	}
	protected boolean[] removeAllAttributes() {
		// remove all attributes
		for (int i = 0; i < this.isEliminatedAttr.length; i++)
			this.isEliminatedAttr[i] = true;
		return isEliminatedAttr;
	}
	
}
