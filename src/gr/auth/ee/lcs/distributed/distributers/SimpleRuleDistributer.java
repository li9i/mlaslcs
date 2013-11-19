/**
 * 
 */
package gr.auth.ee.lcs.distributed.distributers;

import edu.rit.util.Random;
import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.distributed.AbstractRuleDistributer;
import gr.auth.ee.lcs.distributed.IRuleRouter;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * A simple distributer, adding and sending rules with no filtering
 * 
 * @author Miltiadis Allamanis
 * 
 */
public class SimpleRuleDistributer extends AbstractRuleDistributer {

	/**
	 * A simple selector that selects all the rules
	 */
	static IRuleSelector selector = new IRuleSelector() {
		@Override
		public void select(int howManyToSelect, ClassifierSet fromPopulation,
				ClassifierSet toPopulation) {
			toPopulation.merge(fromPopulation);
		}

		@Override
		public void selectWithoutSum(int i, ClassifierSet evolveSet,
				ClassifierSet parents) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void computeFitnessSum(ClassifierSet evolveSet) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double computeFitnessSumNew(ClassifierSet evolveSet) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public void selectWithoutSumNew(int i, ClassifierSet evolveSet,
				ClassifierSet parents, double fitnessSumLocal) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void selectWithoutSumNewSmp(int i, ClassifierSet evolveSet,
				ClassifierSet parents, double fitnessSum, Random prng) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double computeFitnessSumNewSmp(ClassifierSet fromPopulation) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public void selectSmp(int howManyToSelect,
				ClassifierSet fromPopulation, ClassifierSet toPopulation,
				Random prng) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void selectSmp2(int howManyToSelect,
				ClassifierSet fromPopulation, ClassifierSet toPopulation) {
			// TODO Auto-generated method stub
			
		}

	};

	/**
	 * Constructor.
	 * 
	 * @param router
	 *            the router
	 * @param lcs
	 *            the LCS
	 */
	public SimpleRuleDistributer(IRuleRouter router,
			AbstractLearningClassifierSystem lcs, IRuleSelector sendSelector) {
		super(router, lcs, selector, sendSelector);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ILCSMetric#getMetricName()
	 */
	@Override
	public String getMetricName() {
		return "SimpleRuleDistributor";
	}

}
