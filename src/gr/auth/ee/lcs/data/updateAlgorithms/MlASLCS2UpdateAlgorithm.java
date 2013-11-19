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
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.Serializable;

/**
 * An alternative MlASLCS update algorithm.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public class MlASLCS2UpdateAlgorithm extends AbstractUpdateStrategy {

	/**
	 * A data object for the MlASLCS2 update algorithms.
	 * 
	 * @author Miltos Allamanis
	 * 
	 */
	final static class MlASLCSClassifierData implements Serializable {

		/**
		 * 
		 */
		private static final long serialVersionUID = 2584696442026755144L;

		/**
		 * The classifier's fitness
		 */
		public double fitness = .5;

		/**
		 * niche set size estimation.
		 */
		public double ns = 20;

		/**
		 * Match Set Appearances.
		 */
		public int msa = 0;

		/**
		 * true positives.
		 */
		public int tp = 0;

	}

	/**
	 * The theta_del parameter.
	 */
	public static int THETA_DEL = (int) SettingsLoader.getNumericSetting(
			"ASLCS_THETA_DEL", 20);

	/**
	 * The LCS instance being used.
	 * @uml.property  name="myLcs"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Genetic Algorithm.
	 * @uml.property  name="ga"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	public final IGeneticAlgorithmStrategy ga;

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
	 * Number of labels used.
	 * @uml.property  name="numberOfLabels"
	 */
	private final int numberOfLabels;

	/**
	 * The n dumping factor for acc.
	 * @uml.property  name="n"
	 */
	private final double n;

	/**
	 * Constructor.
	 * 
	 * @param lcs
	 *            the LCS being used.
	 * @param labels
	 *            the number of labels
	 * @param geneticAlgorithm
	 *            the GA used
	 * @param nParameter
	 *            the ASLCS dubbing factor
	 * @param fitnessThreshold
	 *            the subsumption fitness threshold to be used.
	 * @param experienceThreshold
	 *            the subsumption experience threshold to be used
	 */
	public MlASLCS2UpdateAlgorithm(final double nParameter,
			final double fitnessThreshold, final int experienceThreshold,
			IGeneticAlgorithmStrategy geneticAlgorithm, int labels,
			AbstractLearningClassifierSystem lcs) {
		this.subsumptionFitnessThreshold = fitnessThreshold;
		this.subsumptionExperienceThreshold = experienceThreshold;
		myLcs = lcs;
		numberOfLabels = labels;
		n = nParameter;
		ga = geneticAlgorithm;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#cover(gr.auth.ee.lcs.classifiers
	 * .ClassifierSet, int)
	 */
	@Override
	public void cover(ClassifierSet population, int instanceIndex) {
		final Classifier coveringClassifier = myLcs
				.getClassifierTransformBridge().createRandomCoveringClassifier(
						myLcs.instances[instanceIndex]);
		population.addClassifier(new Macroclassifier(coveringClassifier, 1),
				false);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObject()
	 */
	@Override
	public Serializable createStateClassifierObject() {
		return new MlASLCSClassifierData();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public double getComparisonValue(Classifier aClassifier, int mode) {
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier
				.getUpdateDataObject();
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			return ((aClassifier.experience < 10) ? 0 : data.fitness);
		case COMPARISON_MODE_DELETION:
			return 1 / (data.fitness * ((aClassifier.experience < THETA_DEL) ? 100.
					: Math.exp(-(Double.isNaN(data.ns) ? 1 : data.ns) + 1)));

		case COMPARISON_MODE_EXPLOITATION:
			final double exploitationFitness = (((double) (data.tp)) / (double) (data.msa));
			return Double.isNaN(exploitationFitness) ? 0 : exploitationFitness;
		default:
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getData(gr.auth.ee.lcs.classifiers
	 * .Classifier)
	 */
	@Override
	public String getData(Classifier aClassifier) {
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier
				.getUpdateDataObject());
		return "internalFitness:" + data.fitness + "tp:" + data.tp + "msa:"
				+ data.msa + "ns:" + data.ns;
	}
	
	@Override
	public void inheritParentParameters(Classifier parentA, Classifier parentB,
			Classifier child) {}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#performUpdate(gr.auth.ee.lcs
	 * .classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void performUpdate(ClassifierSet matchSet, ClassifierSet correctSet) {
		// Nothing here!
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#setComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public void setComparisonValue(Classifier aClassifier, int mode,
			double comparisonValue) {
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier
				.getUpdateDataObject());
		data.fitness = comparisonValue;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.AbstractUpdateStrategy#updateSet(gr.auth.ee.lcs.
	 * classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet, int,
	 * boolean)
	 */
	@Override
	public void updateSet(ClassifierSet population, ClassifierSet matchSet,
			int instanceIndex, boolean evolve) {

		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];

		for (int i = 0; i < numberOfLabels; i++)
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet,
					instanceIndex, i);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		// For each classifier in the matchset
		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i);

			int minCurrentNs = Integer.MAX_VALUE;

			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier
					.getUpdateDataObject();

			for (int l = 0; l < numberOfLabels; l++) {
				// Get classification ability for label l
				final float classificationAbility = cl.myClassifier
						.classifyLabelCorrectly(instanceIndex, l);

				if (classificationAbility == 0)
					continue;

				if (classificationAbility > 0) {
					data.tp += 1;
					final int labelNs = labelCorrectSets[l]
							.getTotalNumerosity();
					
					if (minCurrentNs > labelNs) {
						minCurrentNs = labelNs;
					}
				}
				data.msa += 1;

			}

			cl.myClassifier.experience++;
			if (minCurrentNs != Integer.MAX_VALUE)
				data.ns += .1 * (minCurrentNs - data.ns);
			data.fitness = Math.pow(((double) (data.tp)) / (double) (data.msa),
					n);
			updateSubsumption(cl.myClassifier);
		}

		if (evolve) {
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					ga.evolveSet(labelCorrectSets[l], population, l);
				} else {
					this.cover(population, instanceIndex);
				}
			}
		}

	}

	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @param labelIndex
	 *            the label index
	 * @return the correct set
	 */
	private ClassifierSet generateLabelCorrectSet(final ClassifierSet matchSet,
			final int instanceIndex, final int labelIndex) {
		final ClassifierSet correctSet = new ClassifierSet(null);
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			if (cl.myClassifier.classifyLabelCorrectly(instanceIndex,
					labelIndex) > 0)
				correctSet.addClassifier(cl, false);
		}
		return correctSet;
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
