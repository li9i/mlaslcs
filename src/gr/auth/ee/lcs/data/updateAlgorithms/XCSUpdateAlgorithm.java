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

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;

import java.io.Serializable;

/**
 * The XCS update algorithm.
 * 
 * @author Miltos Allamanis
 */
public final class XCSUpdateAlgorithm extends AbstractUpdateStrategy {

	/**
	 * An object representing the classifier data for the XCS update algorithm.
	 * 
	 * @author Miltos Allamanis
	 */
	public final static class XCSClassifierData implements Serializable {

		/**
		 * Serialization Id.
		 */
		private static final long serialVersionUID = -4348877142305226957L;

		/**
		 * XCS Prediction Error.
		 */
		public double predictionError = 0;

		/**
		 * Action set size
		 */
		public double actionSet = 1;

		/**
		 * Predicted pay-off
		 */
		public double predictedPayOff = 5;

		/**
		 * k XCS parameter
		 */
		public double k;

		/**
		 * XCS Parameter
		 */
		public double fitness = .5;

	}

	/**
	 * XCS learning rate.
	 * @uml.property  name="beta"
	 */
	private final double beta;

	/**
	 * Correct classification payoff.
	 * @uml.property  name="payoff"
	 */
	private final double payoff;

	/**
	 * Accepted Error e0 (accuracy function parameter).
	 * @uml.property  name="e0"
	 */
	private final double e0;

	/**
	 * alpha rate (accuracy function parameter).
	 * @uml.property  name="alpha"
	 */
	private final double alpha;
	/**
	 * The fitness threshold for subsumption.
	 * @uml.property  name="subsumptionFitnessThreshold"
	 */
	private final double subsumptionFitnessThreshold;

	/**
	 * The experience threshold for subsumption.
	 * @uml.property  name="subsumptionExperienceThreshold"
	 */
	private final int subsumptionExperienceThreshold;

	/**
	 * n factor.
	 * @uml.property  name="n"
	 */
	private final double n;

	/**
	 * A double indicating the probability that the GA will run on the matchSet (and not on the correct set).
	 * @uml.property  name="matchSetRunProbability"
	 */
	private final double matchSetRunProbability;

	/**
	 * Genetic Algorithm.
	 * @uml.property  name="ga"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final IGeneticAlgorithmStrategy ga;

	/**
	 * The LCS instance being used.
	 * @uml.property  name="myLcs"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Constructor.
	 * 
	 * @param beta
	 *            the learning rate of the XCS update algorithm
	 * @param P
	 *            the penalty of the XCS update algorithm
	 * @param e0
	 *            the maximum acceptable error for fitness sharing
	 * @param alpha
	 *            used for fitness sharing
	 * @param n
	 *            used for fitness sharing
	 * @param fitnessThreshold
	 *            the fitness threshold for subsumption
	 * @param experienceThreshold
	 *            the experience threshold for subsumption
	 * @param gaMatchSetRunProbability
	 *            the probability of running the GA at the matchSet
	 * @param geneticAlgorithm
	 *            the GA object to be used
	 * @param lcs
	 *            the LCS instance used
	 */
	public XCSUpdateAlgorithm(final double beta, final double P,
			final double e0, final double alpha, final double n,
			final double fitnessThreshold, final int experienceThreshold,
			final double gaMatchSetRunProbability,
			final IGeneticAlgorithmStrategy geneticAlgorithm,
			AbstractLearningClassifierSystem lcs) {
		this.subsumptionFitnessThreshold = fitnessThreshold;
		this.subsumptionExperienceThreshold = experienceThreshold;
		this.beta = beta;
		this.payoff = P;
		this.e0 = e0;
		this.alpha = alpha;
		this.n = n;
		this.ga = geneticAlgorithm;
		this.matchSetRunProbability = gaMatchSetRunProbability;
		myLcs = lcs;
	}

	/**
	 * Calls covering operator.
	 * 
	 * @param population
	 *            the population to cover
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
		return new XCSClassifierData();
	}

	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @return the correct set
	 */
	private ClassifierSet generateCorrectSet(final ClassifierSet matchSet,
			final int instanceIndex) {
		final ClassifierSet correctSet = new ClassifierSet(null);
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < matchSetSize; i++) {
			Macroclassifier cl = matchSet.getMacroclassifier(i);
			if (cl.myClassifier.classifyCorrectly(instanceIndex) == 1)
				correctSet.addClassifier(cl, false);
		}
		return correctSet;
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
		final XCSClassifierData data = ((XCSClassifierData) aClassifier
				.getUpdateDataObject());
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			return data.fitness;
		case COMPARISON_MODE_DELETION:

			return data.k; // TODO: Something else?
		case COMPARISON_MODE_EXPLOITATION:

			return data.predictedPayOff;
		default:
			return 0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#getData(gr.auth
	 * .ee.lcs.classifiers.Classifier)
	 */
	@Override
	public String getData(final Classifier aClassifier) {
		final String response;
		final XCSClassifierData data = ((XCSClassifierData) aClassifier
				.getUpdateDataObject());
		response = "predictionError:" + data.predictionError
				+ ", predictedPayOff:" + data.predictedPayOff;
		return response;
	}
	
	@Override
	public void inheritParentParameters(Classifier parentA, Classifier parentB,
			Classifier child) {}

	/**
	 * Perform Update.
	 * 
	 * @param actionSet
	 *            the action set
	 * @param correctSet
	 *            the correct set
	 */
	@Override
	public void performUpdate(final ClassifierSet actionSet,
			final ClassifierSet correctSet) {
		double accuracySum = 0;

		for (int i = 0; i < actionSet.getNumberOfMacroclassifiers(); i++) {
			Classifier cl = actionSet.getClassifier(i);

			// Get update data object
			XCSClassifierData data = ((XCSClassifierData) cl
					.getUpdateDataObject());
			cl.experience++; // Increase Experience

			double payOff; // the classifier's payoff
			if (correctSet.getClassifierNumerosity(cl) > 0)
				payOff = payoff;
			else
				payOff = 0;

			// Update Predicted Payoff
			if (cl.experience < (1 / beta))
				data.predictedPayOff += (payOff - data.predictedPayOff)
						/ cl.experience;
			else
				data.predictedPayOff += beta * (payOff - data.predictedPayOff);

			// Update Prediction Error
			if (cl.experience < (1 / beta))
				data.predictionError += (Math
						.abs(payOff - data.predictedPayOff) - data.predictionError)
						/ cl.experience;
			else
				data.predictionError += beta
						* (Math.abs(payOff - data.predictedPayOff) - data.predictionError);

			// Update Action Set Estimate
			if (cl.experience < (1 / beta))
				data.actionSet += (actionSet.getTotalNumerosity() - data.actionSet)
						/ cl.experience;
			else
				data.actionSet += beta
						* (actionSet.getTotalNumerosity() - data.actionSet);

			// Fitness Update Step 1
			if (data.predictionError < e0)
				data.k = 1;
			else
				data.k = alpha * Math.pow(data.predictionError / e0, -n);
			accuracySum += data.k * actionSet.getClassifierNumerosity(i);
		}

		// Update Fitness Step 2
		for (int i = 0; i < actionSet.getNumberOfMacroclassifiers(); i++) {
			Classifier cl = actionSet.getClassifier(i);

			// Get update data object
			XCSClassifierData data = ((XCSClassifierData) cl
					.getUpdateDataObject());

			// per micro-classifier
			data.fitness += beta * ((data.k / accuracySum) - data.fitness);
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#setComparisonValue
	 * (gr.auth.ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public void setComparisonValue(final Classifier aClassifier,
			final int mode, final double comparisonValue) {
		final XCSClassifierData data = ((XCSClassifierData) aClassifier
				.getUpdateDataObject());
		data.fitness = comparisonValue; // TODO: Mode changes?

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#updateSet(gr.auth
	 * .ee.lcs.classifiers.ClassifierSet,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet, int)
	 */
	@Override
	public void updateSet(final ClassifierSet population,
			final ClassifierSet matchSet, final int instanceIndex,
			final boolean evolve) {
		/*
		 * Generate correct set
		 */
		final ClassifierSet correctSet = generateCorrectSet(matchSet,
				instanceIndex);

		/*
		 * Cover if necessary
		 */
		if (correctSet.getNumberOfMacroclassifiers() == 0) {
			if (evolve)
				cover(population, instanceIndex);
			return;
		}

		/*
		 * Update
		 */
		performUpdate(matchSet, correctSet);

		/*
		 * Run GA
		 */
		if (evolve) {
			if (Math.random() < matchSetRunProbability)
				ga.evolveSet(matchSet, population, 0);
			else
				ga.evolveSet(correctSet, population, 0);
		}

	}

	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected void updateSubsumption(final Classifier aClassifier) {
		aClassifier
				.setSubsumptionAbility((aClassifier
						.getComparisonValue(COMPARISON_MODE_EXPLOITATION) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold));
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