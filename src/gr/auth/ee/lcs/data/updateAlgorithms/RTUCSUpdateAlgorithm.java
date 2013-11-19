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
package gr.auth.ee.lcs.data.updateAlgorithms;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;

import java.io.Serializable;

/**
 * An alernative Rank and Threshold UCS update algorithm.
 * 
 * @author Miltos Allamanis
 * 
 */
public final class RTUCSUpdateAlgorithm extends AbstractUpdateStrategy {

	/**
	 * A data object for the UCS update algorithm.
	 * 
	 * @author Miltos Allamanis
	 * 
	 */
	final static class UCSClassifierData implements Serializable {

		/**
		 * Serial code for serialization.
		 */
		private static final long serialVersionUID = 3098073593334379507L;

		/**
		 *
		 */
		private double fitness = 1 / 2;

		/**
		 * niche set size estimation.
		 */
		private double cs = 1;

		/**
		 * Match Set Appearances.
		 */
		private int msa = 0;

		/**
		 * true positives.
		 */
		private int tp = 0;

		/**
		 * false positives.
		 */
		private int fp = 0;

		/**
		 * Strength.
		 */
		private double fitness0 = 0;

	}

	/**
	 * Genetic Algorithm.
	 * @uml.property  name="ga"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final IGeneticAlgorithmStrategy ga;

	/**
	 * The \theta_{DEL} parameter of UCS.
	 * @uml.property  name="deleteAge"
	 */
	private final int deleteAge;

	/**
	 * Private variables: the UCS parameter sharing. accuracy0 is considered the subsumption fitness threshold
	 * @uml.property  name="a"
	 */
	private final double a;

	/**
	 * Private variables: the UCS parameter sharing. accuracy0 is considered the subsumption fitness threshold
	 * @uml.property  name="accuracy0"
	 */
	private final double accuracy0;

	/**
	 * Private variables: the UCS parameter sharing. accuracy0 is considered the subsumption fitness threshold
	 * @uml.property  name="n"
	 */
	private final double n;

	/**
	 * Private variables: the UCS parameter sharing. accuracy0 is considered the subsumption fitness threshold
	 * @uml.property  name="b"
	 */
	private final double b;

	/**
	 * A double indicating the probability that the GA will run on the matchSet (and not on the correct set).
	 * @uml.property  name="matchSetRunProbability"
	 */
	private final double matchSetRunProbability;

	/**
	 * The experience threshold for subsumption.
	 * @uml.property  name="subsumptionExperienceThreshold"
	 */
	private final int subsumptionExperienceThreshold;

	/**
	 * A threshold of the classification ability of a classifier in order to be classified as correct (and added to the correct set).
	 * @uml.property  name="correctSetThreshold"
	 */
	private final double correctSetThreshold;

	/**
	 * The number of labels used.
	 * @uml.property  name="numberOfLabels"
	 */
	private final int numberOfLabels;

	/**
	 * The LCS instance being used.
	 * @uml.property  name="myLcs"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Default constructor.
	 * 
	 * @param alpha
	 *            used in fitness sharing
	 * @param nParameter
	 *            used in fitness sharing
	 * @param acc0
	 *            used in fitness sharing: the minimum "good" accuracy
	 * @param learningRate
	 *            the beta of UCS
	 * @param experienceThreshold
	 *            the experience threshold for subsumption
	 * @param gaMatchSetRunProbability
	 *            the probability of running the GA at the matchset
	 * @param geneticAlgorithm
	 *            the genetic algorithm to be used for evolving
	 * @param thetaDel
	 *            the theta del UCS parameter (deletion age)
	 * @param correctSetTheshold
	 *            Threshold the threshold used to set a rule in the correct set
	 * @param lcs
	 *            the LCS instance used
	 */
	public RTUCSUpdateAlgorithm(final double alpha, final double nParameter,
			final double acc0, final double learningRate,
			final int experienceThreshold,
			final double gaMatchSetRunProbability,
			final IGeneticAlgorithmStrategy geneticAlgorithm,
			final int thetaDel, final double correctSetTheshold,
			final int labels, final AbstractLearningClassifierSystem lcs) {
		this.a = alpha;
		this.n = nParameter;
		this.accuracy0 = acc0;
		this.b = learningRate;
		subsumptionExperienceThreshold = experienceThreshold;
		this.matchSetRunProbability = gaMatchSetRunProbability;
		this.ga = geneticAlgorithm;
		deleteAge = thetaDel;
		this.correctSetThreshold = correctSetTheshold;
		numberOfLabels = labels;
		myLcs = lcs;
	}

	/**
	 * Calls covering operator.
	 * 
	 * @param population
	 *            the population where the covering classifier will be added
	 * @param instanceIndex
	 *            the index of the current sample
	 */
	@Override
	public void cover(final ClassifierSet population, final int instanceIndex) {
		final Classifier coveringClassifier = myLcs
				.getClassifierTransformBridge().createRandomCoveringClassifier(
						myLcs.instances[instanceIndex]);
		population.addClassifier(new Macroclassifier(coveringClassifier, 1),
				false);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#
	 * createStateClassifierObject()
	 */
	@Override
	public Serializable createStateClassifierObject() {
		return new UCSClassifierData();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#getComparisonValue
	 * (gr.auth.ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public double getComparisonValue(final Classifier aClassifier,
			final int mode) {
		final UCSClassifierData data = (UCSClassifierData) aClassifier
				.getUpdateDataObject();

		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			final double value = data.fitness
					* ((aClassifier.experience < deleteAge) ? 0 : 1);
			return Double.isNaN(value) ? 0 : value;
		case COMPARISON_MODE_DELETION:

			if (aClassifier.experience < deleteAge) {
				final double result = data.cs / data.fitness;
				return Double.isNaN(result) ? 1 : result;
			}

			return data.cs;

		case COMPARISON_MODE_EXPLOITATION:
			final double acc = (((double) (data.tp)) / (double) (data.msa));
			final double exploitValue = acc
					* ((aClassifier.experience < deleteAge) ? 0 : 1);
			return Double.isNaN(exploitValue) ? 0 : exploitValue;
		default:
			return 0;
		}

	}

	@Override
	public String getData(final Classifier aClassifier) {
		final UCSClassifierData data = ((UCSClassifierData) aClassifier
				.getUpdateDataObject());
		return "tp:" + data.tp;
	}

	@Override
	public void inheritParentParameters(Classifier parentA, Classifier parentB,
			Classifier child) {}
	
	/**
	 * Perform an update to the set.
	 * 
	 * @param matchSet
	 *            the match set used for the update
	 * @param correctSet
	 *            the correct set used for the update
	 */
	@Override
	public void performUpdate(final ClassifierSet matchSet,
			final ClassifierSet correctSet) {
		// Do nothing
	}

	@Override
	public void setComparisonValue(final Classifier aClassifier,
			final int mode, final double comparisonValue) {
		final UCSClassifierData data = ((UCSClassifierData) aClassifier
				.getUpdateDataObject());
		data.fitness = comparisonValue;
	}

	/**
	 * Share a the fitness among a set.
	 * 
	 * @param set
	 *            a correct set to share fitness
	 * @param fitnessToShare
	 *            the numeric value of the fitness to share
	 */
	private void shareFitness(final ClassifierSet set,
								final double fitnessToShare) {
		
		double strengthSum = 0;
		final int setSize = set.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < setSize; i++) {
			Classifier cl = set.getClassifier(i);
			UCSClassifierData data = ((UCSClassifierData) cl.getUpdateDataObject());
			
			cl.experience++;
			data.msa += 1;
			data.tp += 1;
			data.fitness += b * (0 - data.fitness);
			data.cs = data.cs + (b * (setSize - data.cs));
			final double accuracy = ((double) data.tp) / ((double) data.msa);
			if (accuracy > accuracy0) {
				data.fitness0 = 1;

				// Check subsumption
				if (cl.experience >= this.subsumptionExperienceThreshold)
					cl.setSubsumptionAbility(true);

			} else {
				data.fitness0 = a * Math.pow(accuracy / accuracy0, n);
				cl.setSubsumptionAbility(false);
			}

			strengthSum += data.fitness0 * set.getClassifierNumerosity(i);
		}

		// Fix for avoiding problems...
		if (strengthSum == 0)
			strengthSum = 1;

		for (int i = 0; i < setSize; i++) {
			Classifier cl = set.getClassifier(i);
			UCSClassifierData data = ((UCSClassifierData) cl.getUpdateDataObject());
			data.fitness += b * ((data.fitness0 / strengthSum) - data.fitness);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#updateSet(gr.auth
	 * .ee.lcs.classifiers.ClassifierSet,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet) Updates the set setA is the
	 * match set setB is the correct set
	 */
	@Override
	public void updateSet(final ClassifierSet population,
			final ClassifierSet matchSet, final int instanceIndex,
			final boolean evolve) {

		final int[] classifications = myLcs.getClassifierTransformBridge()
				.getDataInstanceLabels(myLcs.instances[instanceIndex]);
		final int numOfCorrectSets = classifications.length;
		final ClassifierSet[] correctSets = new ClassifierSet[numOfCorrectSets];
		for (int i = 0; i < numOfCorrectSets; i++)
			correctSets[i] = new ClassifierSet(null);

		final ClassifierSet wrongSet = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < matchSetSize; i++) {
			final Classifier cl = matchSet.getClassifier(i);
			boolean added = false;
			for (int j = 0; j < numOfCorrectSets; j++) {
				if (cl.classifyLabelCorrectly(instanceIndex, classifications[j]) > 0) {
					correctSets[j].addClassifier(
							matchSet.getMacroclassifier(i), false);
					added = true;
				}
			}
			if (!added)
				wrongSet.addClassifier(matchSet.getMacroclassifier(i), false);
		}

		/*
		 * Cover if necessary
		 */
		if ((wrongSet.getTotalNumerosity() == matchSet.getTotalNumerosity())
				&& evolve) {
			cover(population, instanceIndex);
			return;
		}

		/*
		 * Perform update
		 */
		for (int label = 0; label < numOfCorrectSets; label++) {
			shareFitness(correctSets[label], 1);
		}
		updateWrongSet(wrongSet);

		/*
		 * Run GA
		 */
		if (evolve) {
			if (Math.random() < matchSetRunProbability) {
				ga.evolveSet(matchSet, population, 0);
			} else {
				for (int i = 0; i < numOfCorrectSets; i++)
					if (correctSets[i].getTotalNumerosity() > 0)
						ga.evolveSet(correctSets[i], population, i);
			}
		}

	}

	/**
	 * Update the wrong set classifiers.
	 * 
	 * @param set
	 *            the [!C] set
	 */
	private void updateWrongSet(final ClassifierSet set) {
		final int setSize = set.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			Classifier cl = set.getClassifier(i);
			UCSClassifierData data = ((UCSClassifierData) cl
					.getUpdateDataObject());
			cl.experience++;
			data.msa += 1;
			data.fp += 1;

		}
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
