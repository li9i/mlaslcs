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
/**
 * 
 */
package gr.auth.ee.lcs.geneticalgorithm.selectors;

import edu.rit.util.Random;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

import java.util.Arrays;

/**
 * A tournament selecting the best fitness classifier.
 * 
 * @author Miltos Allamanis
 * 
 */
public class TournamentSelector implements IRuleSelector {

	/**
	 * The size of the tournaments.
	 * @uml.property  name="tournamentSize"
	 */
	private final int tournamentSize;

	/**
	 * The type of the tournaments.
	 * @uml.property  name="max"
	 */
	private final boolean max;

	/**
	 * The percentage of population size, used for tournament selection.
	 * @uml.property  name="percentSize"
	 */
	private final double percentSize;

	/**
	 * The comparison mode for the tournaments.
	 * @uml.property  name="mode"
	 */
	private final int mode;

	/**
	 * Constructor.
	 * 
	 * @param sizeOfTournaments
	 *            the size of the tournament as a percentage of the given set
	 *            size
	 * @param max
	 *            true if the tournament selects the max fitness
	 * @param comparisonMode
	 *            the comparison mode to be used
	 */
	public TournamentSelector(final double sizeOfTournaments,
			final boolean max, final int comparisonMode) {
		this.tournamentSize = 0;
		this.max = max;
		this.mode = comparisonMode;
		percentSize = sizeOfTournaments;
	}

	/**
	 * The default constructor of the selector.
	 * 
	 * @param sizeOfTournaments
	 *            the size of the tournament
	 * @param max
	 *            if true the we select the max fitness participant, else we
	 *            select the min.
	 * @param comparisonMode
	 *            comparison mode @see
	 *            gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy
	 */
	public TournamentSelector(final int sizeOfTournaments, final boolean max,
			final int comparisonMode) {
		this.tournamentSize = sizeOfTournaments;
		this.max = max;
		this.mode = comparisonMode;
		percentSize = 0;
	}

	/**
	 * Select form the population.
	 * 
	 * @param fromPopulation
	 *            the population to select from
	 * @return the index of the selected classifier in the fromPopulation
	 */
	private int select(final ClassifierSet fromPopulation) {
		int size;
		if (tournamentSize == 0) {
			size = (int) Math.floor(fromPopulation.getTotalNumerosity()
					* percentSize);
		} else {
			size = tournamentSize;
		}

		final int[] participants = new int[size];
		// Create random participants
		for (int j = 0; j < participants.length; j++) {
			participants[j] = (int) Math.floor((Math.random() * fromPopulation
					.getTotalNumerosity()));
		}
		return this.tournament(fromPopulation, participants);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.geneticalgorithm.INaturalSelector#select(int,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public final void select(final int howManyToSelect,
			final ClassifierSet fromPopulation, final ClassifierSet toPopulation) {

		for (int i = 0; i < howManyToSelect; i++) {

			toPopulation.addClassifier(
					new Macroclassifier(fromPopulation.getClassifier(this
							.select(fromPopulation)), 1), false);

		}

	}

	/**
	 * Runs a tournament in the fromPopulation with the size defined in
	 * tournamentSize (during construction) The winner of the tournament is
	 * added to the toPopulation.
	 * 
	 * @param fromPopulation
	 *            the source population to run the tournament in
	 * @param participants
	 *            the int[] of indexes of participants
	 * @return the index of the tournament winner
	 */
	public final int tournament(final ClassifierSet fromPopulation,
			final int[] participants) {

		// Sort by order
		Arrays.sort(participants);

		// Best fitness found in tournament
		double bestFitness = max ? Double.NEGATIVE_INFINITY
				: Double.POSITIVE_INFINITY;

		// the index in the participants array of the current competitor
		int currentBestParticipantIndex = -1;

		// The current best participant
		int bestMacroclassifierParticipant = -1;

		// the index of the actual classifier (counting numerosity)
		int currentClassifierIndex = 0;

		// The index of the macroclassifier being used
		int currentMacroclassifierIndex = 0;
		// Run tournament
		do {
			currentBestParticipantIndex += fromPopulation
					.getClassifierNumerosity(currentMacroclassifierIndex);
			while ((currentClassifierIndex < participants.length)
					&& (participants[currentClassifierIndex] <= currentBestParticipantIndex)) {

				// currentParicipant is in this macroclassifier
				final double fitness = fromPopulation.getClassifier(
						currentMacroclassifierIndex).getComparisonValue(mode);

				if ((max ? 1. : -1.) * (fitness - bestFitness) > 0) {
					bestMacroclassifierParticipant = currentMacroclassifierIndex;
					bestFitness = fitness;
				}
				currentClassifierIndex++; // Next!
			}
			currentMacroclassifierIndex++;
		} while (currentClassifierIndex < participants.length);

		if (bestMacroclassifierParticipant >= 0)
			return bestMacroclassifierParticipant;
		else
			return 0;

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
