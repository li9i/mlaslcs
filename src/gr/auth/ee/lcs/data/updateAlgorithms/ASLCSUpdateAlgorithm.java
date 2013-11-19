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
 * The update algorithm for the AS-LCS.
 * 
 * @author Miltos Allamanis
 * 
 */
public final class ASLCSUpdateAlgorithm extends AbstractSLCSUpdateAlgorithm {

	/**
	 * The strictness factor for updating.
	 * @uml.property  name="n"
	 */
	private final double n;

	/**
	 * Object's Constructor.
	 * 
	 * @param nParameter
	 *            the strictness factor ν used in updating
	 * @param fitnessThreshold
	 *            the fitness threshold for subsumption
	 * @param experienceThreshold
	 *            the experience threshold for subsumption
	 * @param gaMatchSetRunProbability
	 *            the probability of running the GA on the match set
	 * @param geneticAlgorithm
	 *            the GA
	 * @param lcs
	 *            the LCS instance used
	 */
	public ASLCSUpdateAlgorithm(final double nParameter,
			final double fitnessThreshold, final int experienceThreshold,
			final double gaMatchSetRunProbability,
			final IGeneticAlgorithmStrategy geneticAlgorithm,
			final AbstractLearningClassifierSystem lcs) {
		super(fitnessThreshold, experienceThreshold, gaMatchSetRunProbability,
				geneticAlgorithm, lcs);
		this.n = nParameter;

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
		final SLCSClassifierData data = (SLCSClassifierData) aClassifier
				.getUpdateDataObject();
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			return data.fitness * ((aClassifier.experience < 8) ? 0 : 1);
		case COMPARISON_MODE_DELETION:
			return 1 / (data.fitness
					* ((aClassifier.experience < 20) ? 100. : Math.exp(-(Double
							.isNaN(data.ns) ? 1 : data.ns) + 1)) * (((aClassifier
					.getCoverage() == 0) && (aClassifier.experience == 1)) ? 0.
					: 1));
			// TODO: Something else?
		case COMPARISON_MODE_EXPLOITATION:
			return data.fitness * ((aClassifier.experience < 8) ? 0 : 1);
			/*
			 * final double exploitationFitness = (((double) (data.tp)) /
			 * (double) (data.msa)); return Double.isNaN(exploitationFitness) ?
			 * .000001 : exploitationFitness;
			 */
		default:
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.updateAlgorithms.AbstractSLCSUpdateAlgorithm#
	 * updateFitness(gr.auth.ee.lcs.classifiers.Classifier, int,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void updateFitness(final Classifier    aClassifier,
							  final int 	      numerosity, 
							  final ClassifierSet correctSet) {
		
		final SLCSClassifierData data = ((SLCSClassifierData) aClassifier.getUpdateDataObject());
		
		
		/* H getClassifierNumerosity() einai sunartisi tis klasis ClassifierSet. 
		 * Kalontas tin apo to correctSet me orisma aClassifier mas leei stin ousia 
		 * (emmesa meso tou ari9mou pou epistrefei)
		 * an o sugkekrimenos classifier brisketai sto correctSet.
		 * */
		if (correctSet.getClassifierNumerosity(aClassifier) > 0) 
			data.tp += 1; // aClassifier at the correctSet
		else // classifier's numerosity = 0
			data.fp += 1;

		// Niche set sharing heuristic...
		
		/*
		 * i fani sti sel 66 to exei allios, to pollaplasiazei epi to numerosity
		 * to exei kapou krumeno. prepei na bre9ei
		 * 
		 * */
		data.fitness = Math.pow(((double) (data.tp)) / (double) (data.msa), n);

	}

	@Override
	public Serializable[] createClassifierObjectArray() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void inheritParentParameters(Classifier parentA, Classifier parentB,
			Classifier child) {}
	
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