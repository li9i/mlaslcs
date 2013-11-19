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
package gr.auth.ee.lcs.geneticalgorithm.selectors;

import edu.rit.util.Random;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * Selects and adds the worst classifier (based on fitness) from the inital
 * ClassifierSet to the target set. It adds the it with
 * howManyToSelect numerosity.
 * 
 * @author Alexandros Philotheou
 * 
 */
public final class WorstClassifierSelector implements IRuleSelector {



	/**
	 * The mode used for comparing classifiers.
	 * @uml.property  name="mode"
	 */
	private final int mode;

	/**
	 * Default constructor.
	 * 
	 * @param maximum
	 *            if by best we mean the max fitness then true, else false
	 * @param comparisonMode
	 *            the mode of the values taken
	 */
	public WorstClassifierSelector(final int comparisonMode) {
		this.mode = comparisonMode;
	}

	/**
	 * Select for population.
	 * 
	 * @param fromPopulation
	 *            the population to select from
	 * @return the index of the best classifier in the set
	 */
	private int select(final ClassifierSet fromPopulation) {
		// Search for the best classifier
		double worstFitness = Double.POSITIVE_INFINITY;
		double worstExp = 0;
		int worstIndex = -1;
		final int popSize = fromPopulation.getNumberOfMacroclassifiers();
		for (int i = 0; i < popSize; i++) {
			final double temp = fromPopulation.getClassifier(i).getComparisonValue(mode);
			
			if ((temp - worstFitness) < 0) {
				worstFitness = temp;
				worstIndex = i;
				worstExp = fromPopulation.getClassifier(i).objectiveCoverage * fromPopulation.getClassifier(i).experience;
			} 
			else if ((Double.compare(temp, worstFitness) == 0) 
					&& (fromPopulation.getClassifier(i).objectiveCoverage * fromPopulation.getClassifier(i).experience <= worstExp)) {
				worstFitness = temp;
				worstIndex = i;
				worstExp = fromPopulation.getClassifier(i).objectiveCoverage * fromPopulation.getClassifier(i).experience;
			}
		}

		return worstIndex;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.geneticalgorithm.INaturalSelector#select(int,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void select(final int howManyToSelect,
			final ClassifierSet fromPopulation, final ClassifierSet toPopulation) {
		// Add it toPopulation
		final int bestIndex = select(fromPopulation);
		if (bestIndex == -1)
			return;
		toPopulation.addClassifier(
									new Macroclassifier(fromPopulation.getClassifier(bestIndex), howManyToSelect), true);
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
	public void selectSmp(int howManyToSelect, ClassifierSet fromPopulation,
			ClassifierSet toPopulation, Random prng) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void selectSmp2(int howManyToSelect, ClassifierSet fromPopulation,
			ClassifierSet toPopulation) {
		// TODO Auto-generated method stub
		
	}

}