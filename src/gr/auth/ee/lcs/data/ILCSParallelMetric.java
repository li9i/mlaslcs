package gr.auth.ee.lcs.data;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;

/**
 * An metrics class interface. This interface will be used for calculating
 * metrics of LCSs. First parallel implementation effort.
 * 
 * @author Vagelis Skartados
 * 
 */
public interface ILCSParallelMetric extends ILCSMetric {
	
	
	public double getMetric(AbstractLearningClassifierSystem lcs, boolean smp);
	
	/**
	 * Evaluate a set of classifiers. Parallel Implementation.
	 * 
	 * @param lcs
	 *            the LCS that we are going to use for evaluation
	 * @return a numeric value indicating ClassifierSet's quality
	 */
	public double getSmpMetric(AbstractLearningClassifierSystem lcs);
	
	
}