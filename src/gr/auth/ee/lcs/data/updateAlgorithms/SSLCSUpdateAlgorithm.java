/*
 *	Copyright (C) 2011 by Allamanis Miltiadis
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */
package gr.auth.ee.lcs.data.updateAlgorithms;

import java.io.Serializable;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;

/**
 * SS-LCS Update Algorithm.
 * 
 * @author Miltos Allamanis
 */
public class SSLCSUpdateAlgorithm extends AbstractSLCSUpdateAlgorithm {

	/**
	 * Reward and penalty percentage parameters.
	 * @uml.property  name="strengthReward"
	 */
	private final double strengthReward;
	/**
	 * Reward and penalty percentage parameters.
	 * @uml.property  name="penalty"
	 */
	private final double penalty;

	/**
	 * Constructor of update algorithm.
	 * 
	 * @param reward
	 *            the reward a correct classifier will be given on correct
	 *            classification
	 * @param penaltyPercent
	 *            the percentage of the reward that the classifier's penalty
	 *            will be when failing to classify
	 * 
	 * @param fitnessThreshold
	 *            the fitness threshold for subsumption
	 * @param experienceThreshold
	 *            the experience threshold for subsumption
	 * 
	 * @param gaMatchSetRunProbability
	 *            the probability to run the GA on the matchset
	 * @param geneticAlgorithm
	 *            the GA to be used for exploration
	 * @param lcs
	 *            the LCS instance used
	 */
	public SSLCSUpdateAlgorithm(final double reward,
			final double penaltyPercent, final double fitnessThreshold,
			final int experienceThreshold, double gaMatchSetRunProbability,
			IGeneticAlgorithmStrategy geneticAlgorithm,
			final AbstractLearningClassifierSystem lcs) {
		super(fitnessThreshold, experienceThreshold, gaMatchSetRunProbability,
				geneticAlgorithm, lcs);
		strengthReward = reward;
		penalty = penaltyPercent;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#getComparisonValue
	 * (gr.auth.ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public final double getComparisonValue(final Classifier aClassifier,
			final int mode) {
		final SLCSClassifierData data = (SLCSClassifierData) aClassifier
				.getUpdateDataObject();

		switch (mode) {
		case COMPARISON_MODE_DELETION:
			// TODO: Something else?
			return data.fitness * ((data.fitness > 0) ? 1 / data.ns : data.ns);
		case COMPARISON_MODE_EXPLOITATION:
			return ((double) data.tp) / ((double) data.msa);
			// return data.str;
		case COMPARISON_MODE_EXPLORATION:
			return data.fitness;
		default:
			return 0;
		}

	}
	
	@Override
	public void inheritParentParameters(Classifier parentA, Classifier parentB,
			Classifier child) {}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.updateAlgorithms.AbstractSLCSUpdateAlgorithm#
	 * updateFitness(gr.auth.ee.lcs.classifiers.Classifier, int,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public final void updateFitness(final Classifier aClassifier,
			final int numerosity, final ClassifierSet correctSet) {
		final SLCSClassifierData data = ((SLCSClassifierData) aClassifier
				.getUpdateDataObject());
		if (Double.isNaN(data.str) || Double.isInfinite(data.str))
			data.str = 0;
		if (correctSet.getClassifierNumerosity(aClassifier) > 0) {
			// aClassifier belongs to correctSet
			data.str += strengthReward / correctSet.getTotalNumerosity();
			data.ns = (data.ns * data.tp + correctSet.getTotalNumerosity())
					/ ((double) (data.tp + 1.));
			data.tp++;
		} else {
			data.fp++;
			final double punishment = penalty * strengthReward / (data.ns);
			data.str -= (Double.isNaN(punishment) || Double
					.isInfinite(punishment)) ? penalty * strengthReward
					: punishment;
		}

		data.fitness = (data.str / data.msa);
		if (Double.isNaN(data.fitness))
			data.fitness = Double.MIN_VALUE;
	}

	@Override
	public Serializable[] createClassifierObjectArray() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void updateSetNew(ClassifierSet population,
			ClassifierSet matchSet, int instanceIndex, boolean evolve){}

	@Override
	public void coverSmp(ClassifierSet population, int instanceIndex) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateSetSmp(ClassifierSet population, ClassifierSet matchSet,
			int instanceIndex, boolean evolve) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateSetNewSmp(ClassifierSet population,
			ClassifierSet matchSet, int instanceIndex, boolean evolve) {
		// TODO Auto-generated method stub
		
	};

}