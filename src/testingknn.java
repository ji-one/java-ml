
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Method;

//201724472 
public class testingknn {
	
	
	private DataSet dataset ;
	private Strategy strategy ;
	private DataSetInput DataInput ;
	private StepwiseVariableSelection backward;
	
	@Before
	public void settingbackward() throws Exception{
		DataInput = new FileInput("./data/census");
		dataset = new BinaryDataSet(DataInput);
		strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		backward = new backwardsElimination(dataset, strategy);
	}
	/*
	 * Purpose: construct backwardElimination
	 * Input:  backwardElimination make  backwardElimination object 
	 * Expected: 
	 * 		dataset = getDataSet()
	 * 		strategy = getStrategy()
	 * 
	 * 
	 */
	@Test
	public void testbackward(){

		assertEquals(dataset, backward.getDataSet());
		assertEquals(strategy, backward.getStrategy());
	}

	/*
	 * Purpose: distance between two vectors 
	 * Input: getDistance vector1 = {1,1,1,1} and vector2 = {0,0,0,0} -> 4
	 * Expected: 
	 * 		return 4
	 */
	@Test
	public void testGetDistance() throws Exception {

		Method method = backward.getClass().getDeclaredMethod("getDistance", int[].class, int[].class);
		method.setAccessible(true);

		int[] vector1 = {1,1,1,1};
		int[] vector2 = {0,0,0,0};
		
		boolean[] isEliminatedAttr = new boolean[4];
		backward.setIsEliminatedAttr(isEliminatedAttr);
		
		double distance = (double) method.invoke(backward, vector1, vector2);
		assertEquals(distance, 4, 0.0001);
	}


	
	/*
	 * Purpose: voteCount Test
	 * Input: voteCount / indices {1,0,2} -> 1
	 * Expected:
	 * 		return 1
	 */
	@Test
	public void testVoteCount() throws Exception {

		
		Method method = backward.getClass().getDeclaredMethod("voteCount", int[].class);
		method.setAccessible(true);
		
		double[] Weights = new double[9];
		for (int i = 0; i < Weights.length; i++)
			Weights[i] = 1.0;
		
		backward.setInstanceWeights(Weights);
		backward.setkOpt(3);
		
		int[] indices = {1,0,2};
		int vote = (int)method.invoke(backward, indices);
		assertEquals(vote, 1);
	}
	/*
	 * Purpose: check distance setting
	 * Input: setCrossValidationDistancefortest null-> {{0,Double.POSITIVE_INFINITY},{Double.POSITIVE_INFINITY,0}}
	 * Expected:
	 * 		return {{0,Double.POSITIVE_INFINITY},{Double.POSITIVE_INFINITY,0}}
	 */
	@Test
	public void testdistancesetting() throws Exception {

		
		Method method = backward.getClass().getDeclaredMethod("setCrossValidationDistancefortest");
		method.setAccessible(true);
		
		double[][] expect = {{0,Double.POSITIVE_INFINITY},{Double.POSITIVE_INFINITY,0}};
		double[][] distance = (double[][]) method.invoke(backward);
		assertArrayEquals(distance, expect);
	}
	
	
	
	/*
	 * Purpose: check attribute removed
	 * Input: removeAllAttributes null-> {true,true,.....,true}
	 * Expected:
	 * 		return {true,true,.....,true}
	 */
	@Test
	public void testremoveattribute() throws Exception {

		
		Method method = backward.getClass().getDeclaredMethod("removeAllAttributes");
		method.setAccessible(true);
		
		boolean[] isEliminatedAttr = new boolean[dataset.numAttrs];
		backward.setIsEliminatedAttr(isEliminatedAttr);

		boolean[] removeattribute = new boolean[dataset.numAttrs];
		for(int i=0;i<dataset.numAttrs;i++)
			removeattribute[i] = true;
		boolean[] checkifremoved = (boolean[]) method.invoke(backward);
		
		assertArrayEquals(checkifremoved, removeattribute);
	}
	
}
