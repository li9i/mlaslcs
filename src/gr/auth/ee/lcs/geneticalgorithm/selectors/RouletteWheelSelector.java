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

import edu.rit.pj.IntegerForLoop;
import edu.rit.pj.ParallelRegion;
import edu.rit.pj.ParallelSection;
import edu.rit.pj.ParallelTeam;
import edu.rit.util.Random;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * A Natural Selection operator performing a weighted roulette wheel selection.
 * This implementation contracts that all classifier have positive values of
 * fitness. TODO: Throw exception otherwise?
 * 
 * @author Miltos Allamanis
 * 
 */

public class RouletteWheelSelector implements IRuleSelector {

	/**
	 * The comparison mode used for fitness selecting.
	 * @uml.property  name="mode"
	 */
	private final int mode;

	/**
	 * Private variable for selecting maximum or minimum selection.
	 * @uml.property  name="max"
	 */
	private final boolean max;
	
	private double fitnessSum;
	
	private static int modeSmp;
	
	private static boolean maxSmp;
	
	public ParallelTeam ptComputeFitnessSum;
	
	static double fitnessSumSmp;
	
	static ClassifierSet fromPopulationSmp;

	/**
	 * Constructor.
	 * 
	 * @param comparisonMode
	 *            the comparison mode
	 * @param max
	 *            whether the selector selects min or max fitness (when max,
	 *            max=true)
	 */
	public RouletteWheelSelector(final int comparisonMode, 
								  final boolean max) {
		
		mode = comparisonMode;
		this.max = max;
		
		if (mode ==2)
		{
			modeSmp = comparisonMode;
			maxSmp = max;
		}	
		
		ptComputeFitnessSum = new ParallelTeam();
	}
	
	@Override
	public final void computeFitnessSum(final ClassifierSet fromPopulation)
	{
		final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		if (mode == AbstractUpdateStrategy.COMPARISON_MODE_DELETION) {}
		
		
		fitnessSum = 0;
		for (int i = 0; i < numberOfMacroclassifiers; i++) {

			final double fitnessValue = fromPopulation.getClassifierNumerosity(i)
					* fromPopulation.getClassifier(i).getComparisonValue(mode);
			fitnessSum += max ? fitnessValue : 1 / (fitnessValue + Double.MIN_NORMAL);
		}
	}
	
	@Override
	public final double computeFitnessSumNew(final ClassifierSet fromPopulation)
	{
		final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		double fitnessSumLocal = 0;
		
		for (int i = 0; i < numberOfMacroclassifiers; i++) {

			final double fitnessValue = fromPopulation.getClassifierNumerosity(i)
					* fromPopulation.getClassifier(i).getComparisonValue(mode);
			fitnessSumLocal += max ? fitnessValue : 1 / (fitnessValue + Double.MIN_NORMAL);
		}
		
		return fitnessSumLocal;
	}
	
	@Override
	public final double computeFitnessSumNewSmp(final ClassifierSet fromPopulation)
	{
		final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		double fitnessSumLocal = 0;
		
		for (int i = 0; i < numberOfMacroclassifiers; i++) {

			final double fitnessValue = fromPopulation.getClassifierNumerosity(i)
					* fromPopulation.getClassifier(i).getComparisonValue(modeSmp);
			fitnessSumLocal += maxSmp ? fitnessValue : 1 / (fitnessValue + Double.MIN_NORMAL);
		}
		
		return fitnessSumLocal;
	}
	
	@Override
	public final void selectWithoutSum  (final int howManyToSelect,
										   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
										   final ClassifierSet toPopulation) {
		
		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = Math.random() * fitnessSum;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette
	}
	
	@Override
	public final void selectWithoutSumNew(final int howManyToSelect,
			   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
			   final ClassifierSet toPopulation,
			   final double fitnessSumLocal) {
		
		
		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = Math.random() * fitnessSumLocal;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette

	}
	
	@Override
	public final void selectWithoutSumNewSmp(final int howManyToSelect,
			   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
			   final ClassifierSet toPopulation,
			   final double fitnessSum,
			   final Random prng) {

		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = prng.nextDouble() * fitnessSum;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette
	}

	/**
	 * Roulette Wheel selection strategy.
	 * 
	 * @param howManyToSelect
	 *            the number of draws.
	 * @param fromPopulation
	 *            the ClassifierSet from which the selection will take place
	 * @param toPopulation
	 *            the ClassifierSet to which the selected Classifiers will be
	 *            added
	 */
	@Override
	public final void select(final int howManyToSelect,
							   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
							   final ClassifierSet toPopulation) {
		
		
		final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		if (mode == AbstractUpdateStrategy.COMPARISON_MODE_DELETION) {}
		
		// Find total sum
		double fitnessSum = 0;
		for (int i = 0; i < numberOfMacroclassifiers; i++) {

			final double fitnessValue = fromPopulation.getClassifierNumerosity(i)
					* fromPopulation.getClassifier(i).getComparisonValue(mode);
			fitnessSum += max ? fitnessValue : 1 / (fitnessValue + Double.MIN_NORMAL);
		}
		
		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = Math.random() * fitnessSum;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette

	}
	
	@Override
	public final void selectSmp(final int howManyToSelect,
							   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
							   final ClassifierSet toPopulation, Random prng) {
	
		final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		if (mode == AbstractUpdateStrategy.COMPARISON_MODE_DELETION) {}
		
		// Find total sum
		double fitnessSum = 0;
		for (int i = 0; i < numberOfMacroclassifiers; i++) {

			final double fitnessValue = fromPopulation.getClassifierNumerosity(i)
					* fromPopulation.getClassifier(i).getComparisonValue(mode);
			fitnessSum += max ? fitnessValue : 1 / (fitnessValue + Double.MIN_NORMAL);
		}
		
		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = prng.nextDouble() * fitnessSum;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette
		
	}
	
	@Override
	public final void selectSmp2(final int howManyToSelect,
							   final ClassifierSet fromPopulation, // to correctSet(evolve) i o population(addClassifier) i delete
							   final ClassifierSet toPopulation) {
	
		//final int numberOfMacroclassifiers = fromPopulation.getNumberOfMacroclassifiers();
		
		fromPopulationSmp = fromPopulation;
		
		if (mode == AbstractUpdateStrategy.COMPARISON_MODE_DELETION) {}
		
		try{
			ptComputeFitnessSum.execute( new ParallelRegion() {
			
				public void start()
				{
					fitnessSumSmp = 0;
				}
			
				public void run() throws Exception
				{
					execute(0, fromPopulationSmp.getNumberOfMacroclassifiers()-1, new IntegerForLoop() {
						
						double fitnessSum_thread;
						
						public void start()
						{
							fitnessSum_thread = 0;
						}						
						public void run(int first, int last)
						{
							for ( int i = first; i <= last ; ++i )
							{
								fitnessSum_thread += fromPopulationSmp.getClassifierNumerosity(i)
												*fromPopulationSmp.getClassifier(i).getComparisonValue(mode);
							}
						}
						
						public void finish() throws Exception
						{
							region().critical( new ParallelSection() {
								public void run()
								{
									fitnessSumSmp += fitnessSum_thread;
								}
							});
						}
						
					});
				}	
			});
		}
		catch ( Exception e)
		{
			e.printStackTrace();
		}
		
		// Repeat roulette for howManyToSelect times
		for (int i = 0; i < howManyToSelect; i++) {
			// Roulette
			final double rand = Math.random() * fitnessSumSmp;

			double tempSum = 0;
			int selectedIndex = -1;

			do {
				selectedIndex++;
				final double tempValue = fromPopulation.getClassifierNumerosity(selectedIndex)
						 * fromPopulation.getClassifier(selectedIndex).getComparisonValue(mode);

				tempSum += max ? tempValue : 1 / (tempValue + Double.MIN_NORMAL);
				
			} while (tempSum < rand);
			// Add selectedIndex
			toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(selectedIndex), 1), false);

		} // next roulette
		
	}
	
}