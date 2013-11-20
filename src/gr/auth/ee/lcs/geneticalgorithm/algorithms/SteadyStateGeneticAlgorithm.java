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
package gr.auth.ee.lcs.geneticalgorithm.algorithms;

import java.util.Vector;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IBinaryGeneticOperator;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;
import gr.auth.ee.lcs.geneticalgorithm.IUnaryGeneticOperator;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import edu.rit.util.Random;
import edu.rit.pj.ParallelTeam;
import edu.rit.pj.IntegerForLoop;
import edu.rit.pj.ParallelSection;
import edu.rit.pj.ParallelRegion;

/**
 * A steady-stage GA that selects two individuals from a set (with probability
 * proportional to their total fitness) and performs a crossover and mutation,
 * corrects the classifier (if needed) and adds it to the set.
 * 
 * @author Miltos Allamanis
 * 
 */
public class SteadyStateGeneticAlgorithm implements IGeneticAlgorithmStrategy {
	
	public class EvolutionOutcome {
		
		public Vector<Integer> indicesToSubsume;
		public ClassifierSet newClassifierSet;
		public long subsumptionTime;
		public long selectionTime;
		
		public long timeA;
		public long timeB;
		public long timeC;
		public long timeD;
	}

	/**
	 * The selector used for the next generation selection.
	 */
	final private IRuleSelector gaSelector;

	/**
	 * The crossover operator that will be used by the GA.
	 */
	final private IBinaryGeneticOperator crossoverOp;

	/**
	 * The mutation operator used by the GA.
	 */
	final private IUnaryGeneticOperator mutationOp;

	/**
	 * The GA activation age. The population must have an average age, greater that this in order for the GA to run.
	 */
	private int gaActivationAge;

	/**
	 * The current timestamp. Used by the GA to count generations.
	 */
	private int timestamp = 0;
	
	private final int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels",1);

	
	private int[] timestamps = new int[numberOfLabels];

	/**
	 * The rate that the crossover is performed.
	 */
	private final float crossoverRate;

	/**
	 * The number of children per generation.
	 */
	private static final int CHILDREN_PER_GENERATION = 2;

	/**
	 * The LCS instance being used.
	 */
	private final AbstractLearningClassifierSystem myLcs;
	
	Vector<Integer> indicesToSubsume;
	
	ClassifierSet newClassifiersSet;
	
	private long subsumptionTime;
	
	private long selectionTime;
	
	private int evolutionConducted;
	
	private int numberOfSubsumptions;
	
	private int numberOfNewClassifiers;
	
	private int numberOfDeletions;
	
	private long deletionTime; 
	
	
	private final boolean THOROUGHLY_CHECK_WITH_POPULATION = SettingsLoader.getStringSetting("THOROUGHLY_CHECK_WITH_POPULATION", "true").equals("true");
	
	static final boolean THOROUGHLY_CHECK_WITH_POPULATION_SMP = SettingsLoader.getStringSetting("THOROUGHLY_CHECK_WITH_POPULATION", "true").equals("true");;
	
	private final boolean gaPerLabel = SettingsLoader.getStringSetting("gaPerLabel", "true").equals("true");

	

	private final int iterations = (int) SettingsLoader.getNumericSetting("trainIterations",1000);
	
	ParallelTeam ptEvolveOneLabel;
	
	ParallelTeam ptSubsume;
	
	ParallelTeam ptSelect;
	
	static long seedSmp;
	
	static Classifier parentASmp;
	
	static Classifier parentBSmp;
	
	static ClassifierSet populationSmp;
	
	static Vector<Integer> indicesToSubsumeSmp;
	
	static ClassifierSet newClassifiersSetSmp;
	
	static IRuleSelector gaSelectorSmp;
	
	static IBinaryGeneticOperator crossoverOpSmp;
	
	static IUnaryGeneticOperator mutationOpSmp;
	
	static float crossoverRateSmp;
	
	static AbstractLearningClassifierSystem myLcsSmp;	
	
	static long subsumptionTimeSmp;
	
	static double fitnessSumSmp;
	
	static ClassifierSet evolveSetSmp;
	
	public int crossoverOperator = (int) SettingsLoader.getNumericSetting("crossoverOperator", 0);;
	public static final int SINGLEPOINT_CROSSOVER = 0;
	public static final int MULTIPOINT_CROSSOVER = 1;

	
	/**
	 * Parents' subsumption method.
	 * @param population
	 * 			the rule population
	 * @param parentA
	 * 			parent #1
	 * @param indexA
	 * 			the position of parentA as a macroclassifier inside the myClassifiers vector
	 * @param parentB
	 * 			parent #2
	 * @param indexB
	 * 			the position of parentB as a macroclassifier inside the myClassifiers vector
	 * @param child
	 * 			the child produced by the GA
	 * 
	 * @author alexandros filotheou
	 */
	public boolean letParentsSubsume (ClassifierSet population, 
										Classifier parentA,
										Classifier parentB,
										final Classifier child) {

		final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
		
		// find the indices of the parents, inside the myMacroclassifiers vector.
		int indexA = -1;
		int indexB = -1;
		
		for (int i = 0; i < population.getNumberOfMacroclassifiers(); i++){
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentA.getSerial()) 
				indexA = i;
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentB.getSerial()) 
				indexB = i;
		}
		
		Classifier subsumer = null;	
		int index = -1;
		
		boolean parentACanSubsume = (parentA.canSubsume() && parentA.isMoreGeneral(child)) || parentA.equals(child);
		double fitnessA = parentA.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceA = parentA.experience;
		
		boolean parentBCanSubsume = (parentB.canSubsume() && parentB.isMoreGeneral(child)) || parentB.equals(child);
		double fitnessB = parentB.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceB = parentB.experience;
		
		
		
		if (indexA >= 0 && indexB >= 0) {
			if (indexA != indexB) {
				if (parentACanSubsume && parentBCanSubsume) {
					subsumer = (fitnessA > fitnessB) ? parentA : parentB;
					if (fitnessA == fitnessB) subsumer = (experienceA > experienceB) ? parentA : parentB;

					index = (subsumer.equals(parentA)) ? indexA : indexB;
				}
				else if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
				else if (parentBCanSubsume) {
					subsumer = parentB;
					index = indexB;
				}
			}
			else // indexA = indexB 
				if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
		}
		// indexA < 0 OR indexB < 0
		else if (indexA >= 0) {
			if (parentACanSubsume) {
				subsumer = parentA;
				index = indexA;
			}
		}
		else if (indexB >= 0) {
			if (parentBCanSubsume) {
				subsumer = parentB;
				index = indexB;
			}
		}	
		
		if (subsumer != null) {
				population.getMacroclassifiersVector().get(index).numerosity++;
				population.getMacroclassifiersVector().get(index).numberOfSubsumptions++;
				population.totalNumerosity++;
				theControlStrategy.controlPopulation(population);
				return true;
		}
		return false;
	} 
	

	
	
	
	
	
	
	public boolean letParentsSubsumeSmp(ClassifierSet population, 
											Classifier parentA,
											Classifier parentB,
											final Classifier child) {
		
		final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
		
		// find the indices of the parents, inside the myMacroclassifiers vector.
		int indexA = -1;
		int indexB = -1;
		
		for (int i = 0; i < population.getNumberOfMacroclassifiers(); i++){
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentA.getSerial()) 
				indexA = i;
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentB.getSerial()) 
				indexB = i;
		}
		
		Classifier subsumer = null;	
		int index = -1;
		
		boolean parentACanSubsume = (parentA.canSubsume() && parentA.isMoreGeneral(child)) || parentA.equals(child);
		double fitnessA = parentA.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceA = parentA.experience;
		
		boolean parentBCanSubsume = (parentB.canSubsume() && parentB.isMoreGeneral(child)) || parentB.equals(child);
		double fitnessB = parentB.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceB = parentB.experience;
		
		
		
		if (indexA >= 0 && indexB >= 0) {
			if (indexA != indexB) {
				if (parentACanSubsume && parentBCanSubsume) {
					subsumer = (fitnessA > fitnessB) ? parentA : parentB;
					if (fitnessA == fitnessB) subsumer = (experienceA > experienceB) ? parentA : parentB;

					index = (subsumer.equals(parentA)) ? indexA : indexB;
				}
				else if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
				else if (parentBCanSubsume) {
					subsumer = parentB;
					index = indexB;
				}
			}
			else // indexA = indexB 
				if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
		}
		// indexA < 0 OR indexB < 0
		else if (indexA >= 0) {
			if (parentACanSubsume) {
				subsumer = parentA;
				index = indexA;
			}
		}
		else if (indexB >= 0) {
			if (parentBCanSubsume) {
				subsumer = parentB;
				index = indexB;
			}
		}	
		
		if (subsumer != null) {
				population.getMacroclassifiersVector().get(index).numerosity++;
				population.getMacroclassifiersVector().get(index).numberOfSubsumptions++;
				population.totalNumerosity++;
				theControlStrategy.controlPopulationSmp(population);
				return true;
		}
		
		return false;
	}
	
	
	
	
	
	private int letParentsSubsumeNew (ClassifierSet population, 
										Classifier parentA,
										Classifier parentB,
										final Classifier child) {
		
		final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
		
		// find the indices of the parents, inside the myMacroclassifiers vector.
		int indexA = -1;
		int indexB = -1;
		
		for (int i = 0; i < population.getNumberOfMacroclassifiers(); i++){
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentA.getSerial()) 
				indexA = i;
			if (population.getMacroclassifiersVector().get(i).myClassifier.getSerial() == parentB.getSerial()) 
				indexB = i;
		}

		
		Classifier subsumer = null;	
		int index = -1;
		
		boolean parentACanSubsume = (parentA.canSubsume() && parentA.isMoreGeneral(child)) || parentA.equals(child);
		double fitnessA = parentA.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceA = parentA.experience;
		
		boolean parentBCanSubsume = (parentB.canSubsume() && parentB.isMoreGeneral(child)) || parentB.equals(child);
		double fitnessB = parentB.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		int experienceB = parentB.experience;
		
		
		
		if (indexA >= 0 && indexB >= 0) {
			if (indexA != indexB) {
				if (parentACanSubsume && parentBCanSubsume) {
					subsumer = (fitnessA > fitnessB) ? parentA : parentB;
					if (fitnessA == fitnessB) subsumer = (experienceA > experienceB) ? parentA : parentB;

					index = (subsumer.equals(parentA)) ? indexA : indexB;
				}
				else if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
				else if (parentBCanSubsume) {
					subsumer = parentB;
					index = indexB;
				}
			}
			else // indexA = indexB 
				if (parentACanSubsume) {
					subsumer = parentA;
					index = indexA;
				}
		}
		// indexA < 0 OR indexB < 0
		else if (indexA >= 0) {
			if (parentACanSubsume) {
				subsumer = parentA;
				index = indexA;
			}
		}
		else if (indexB >= 0) {
			if (parentBCanSubsume) {
				subsumer = parentB;
				index = indexB;
			}
		}	
		
		if (subsumer != null) {
				return index;
		}
		return -1;
		
	}
	
	
	
	
	/**
	 * Default constructor.
	 * 
	 * @param gaSelector
	 *            the INautralSelector that selects parents for next generation
	 * @param crossoverOperator
	 *            the crossover operator that will be used
	 * @param mutationOperator
	 *            the mutation operator that will be used
	 * @param gaActivationAge
	 *            the age of the population that activates the G.A.
	 * @param crossoverRate
	 *            the rate at which the crossover operator will be called
	 * @param lcs
	 *            the LCS instance used
	 * 
	 */
	public SteadyStateGeneticAlgorithm (final IRuleSelector gaSelector,
										final IBinaryGeneticOperator crossoverOperator,
										final float crossoverRate,
										final IUnaryGeneticOperator mutationOperator,
										final int gaActivationAge,
										final AbstractLearningClassifierSystem lcs) {
		
		this.gaSelector = gaSelector;
		this.crossoverOp = crossoverOperator;
		this.mutationOp = mutationOperator;
		this.gaActivationAge = gaActivationAge;
		this.crossoverRate = crossoverRate;
		this.myLcs = lcs;
		
		
		crossoverRateSmp = crossoverRate;
		
		gaSelectorSmp = this.gaSelector;
		
		crossoverOpSmp = crossoverOperator;
		mutationOpSmp = mutationOperator;
		myLcsSmp = lcs;
		
		ptEvolveOneLabel = new ParallelTeam();
		
		ptSubsume = new ParallelTeam();
		
		ptSelect = new ParallelTeam();
		
		
	}




	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy#evolveSet(gr
	 * .auth.ee.lcs.classifiers.ClassifierSet,
	 * gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public final void evolveSet(final ClassifierSet evolveSet,
								  final ClassifierSet population,
								  final int label) {

		timestamp++;
		
		evolutionConducted = 0;
		subsumptionTime = 0;
		numberOfSubsumptions = 0;
		numberOfNewClassifiers = 0;
		numberOfDeletions = 0;
		deletionTime = 0;

		
		final int meanAge = getMeanAge(evolveSet); 
		
		if (timestamp - meanAge < this.gaActivationAge) {
			return;
		}
		
		evolutionConducted = 1;

		final int evolveSetSize = evolveSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < evolveSetSize; i++) {
			evolveSet.getClassifier(i).timestamp = timestamp;			
		}

		final ClassifierSet parents = new ClassifierSet(null);

		// Select parents
		gaSelector.select(1, evolveSet, parents); 
		final Classifier parentA = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		gaSelector.select(1, evolveSet, parents);
		final Classifier parentB = parents.getClassifier(0);
		parents.deleteClassifier(0);
		

		// Reproduce
		for (int i = 0; i < CHILDREN_PER_GENERATION; i++) {
			
			Classifier child;
			
			boolean proceedMyChild = false;
			
			
			// produce a child
			if (Math.random() < crossoverRate && !parentA.equals(parentB)) {
				
				
				int chromosomeSize = -1;
				if (crossoverOperator == MULTIPOINT_CROSSOVER)
					chromosomeSize = parentA.size() - 2 * (numberOfLabels - 1); 
				else if (crossoverOperator == SINGLEPOINT_CROSSOVER)
					chromosomeSize = parentA.size();
				/*
				 * The point at which the crossover will occur
				 */
				int mutationPoint = (int) Math.round(Math.random() * chromosomeSize - 1);
				
				child = crossoverOp.operate((i == 0) ? parentB : parentA, (i == 0) ? parentA : parentB, label, mutationPoint);
				
			} 
			
			else {
				child = (Classifier) ((i == 0) ? parentA : parentB).clone();
				child.setComparisonValue(
						AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION,
						((i == 0) ? parentA : parentB)
								.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
			}
			
			child = mutationOp.operate(child);
			
			// 0-coverage prevention. every child introduced in the population will be non 0-coverage.
			for (int ins = 0; ins < myLcs.instances.length; ins++) {
				if (child.isMatch(myLcs.instances[ins])) {
					proceedMyChild = true;
					break;
				}
			}
			
			
			if (proceedMyChild) {
			
				child.inheritParametersFromParents(parentA, parentB); 
				
				myLcs.getClassifierTransformBridge().fixChromosome(child);
				
				child.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_GA);
				child.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
	
				child.created = myLcs.totalRepetition; //timestamp; 
				
				long time1 = -System.currentTimeMillis();
				
				//check subsumption by parents
				boolean parentsSubsumed = letParentsSubsume(population, parentA, parentB, child);
				if (!parentsSubsumed) {	
					// parents couldn't subsume, should i check with the population?
					population.addClassifier(new Macroclassifier(child, 1), THOROUGHLY_CHECK_WITH_POPULATION);
				
					if (population.subsumed)
						numberOfSubsumptions++;
					else
						numberOfNewClassifiers++;
					
				}
				else
					numberOfSubsumptions++;
				
				time1 += System.currentTimeMillis();
				
				subsumptionTime += time1;
				
				deletionTime += population.getPopulationControlStrategy().getDeletionTime();
				
				numberOfDeletions += population.getPopulationControlStrategy().getNumberOfDeletionsConducted(); 
				
		
			
			subsumptionTime -= deletionTime;
			}
		}
	}
	
	
	@Override
	public final void evolveSetSmp (final ClassifierSet evolveSet,
									  final ClassifierSet population,
									  final int label) {
		timestamp++;
		
		evolutionConducted = 0;
		subsumptionTime = 0;
		numberOfSubsumptions = 0;
		numberOfNewClassifiers = 0;
		numberOfDeletions = 0;
		deletionTime = 0;

		
		final int meanAge = getMeanAge(evolveSet); 
		
		if (timestamp - meanAge < this.gaActivationAge) {
			return;
		}
		
		evolutionConducted = 1;

		final int evolveSetSize = evolveSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < evolveSetSize; i++) {
			evolveSet.getClassifier(i).timestamp = timestamp;			
		}

		final ClassifierSet parents = new ClassifierSet(null);

		// Select parents
		gaSelector.computeFitnessSum(evolveSet);
		
		gaSelector.selectWithoutSum(1, evolveSet, parents); 
		final Classifier parentA = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		gaSelector.selectWithoutSum(1, evolveSet, parents);
		final Classifier parentB = parents.getClassifier(0);
		parents.deleteClassifier(0);
		

		// Reproduce
		for (int i = 0; i < CHILDREN_PER_GENERATION; i++) {
			Classifier child;
			// produce a child
			if (Math.random() < crossoverRate && !parentA.equals(parentB)) {
				
				int chromosomeSize = -1;
				if (crossoverOperator == MULTIPOINT_CROSSOVER)
					chromosomeSize = parentA.size() - 2 * (numberOfLabels - 1); 
				else if (crossoverOperator == SINGLEPOINT_CROSSOVER)
					chromosomeSize = parentA.size();
				/*
				 * The point at which the crossover will occur
				 */
				int mutationPoint = (int) Math.round(Math.random() * chromosomeSize - 1);
				
				child = crossoverOp.operate((i == 0) ? parentB : parentA, (i == 0) ? parentA : parentB, label, mutationPoint);
				
			} 
			else {
				child = (Classifier) ((i == 0) ? parentA : parentB).clone();
				child.setComparisonValue(
						AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION,
						((i == 0) ? parentA : parentB)
								.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
			}

			child = mutationOp.operate(child);
			
			child.inheritParametersFromParents(parentA, parentB);
			
			myLcs.getClassifierTransformBridge().fixChromosome(child);
			//System.out.println("child after: " + child);
			
			child.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_GA);
			
			child.created = myLcs.totalRepetition; //timestamp; 
			
			long time1 = -System.currentTimeMillis();
			
			//check subsumption by parents
			boolean parentsSubsumed = letParentsSubsumeSmp(population, parentA, parentB, child);
			if (!parentsSubsumed) {	
				// parents couldn't subsume, should i check with the population?
				population.addClassifierSmp(new Macroclassifier(child, 1), THOROUGHLY_CHECK_WITH_POPULATION, ptSubsume);
			
				if (population.subsumed)
					numberOfSubsumptions++;
				else
					numberOfNewClassifiers++;
				
			}
			else
				numberOfSubsumptions++;
			
			time1 += System.currentTimeMillis();
			
			subsumptionTime += time1;
			
			deletionTime += population.getPopulationControlStrategy().getDeletionTime();
			
			numberOfDeletions += population.getPopulationControlStrategy().getNumberOfDeletionsConducted(); 
			
		}
		
		subsumptionTime -= deletionTime;
	}
	
	
	
	
	@Override
	public final void evolveSetNew (final ClassifierSet evolveSet,
								  		final ClassifierSet population,
								  		int label) {
	
		subsumptionTime = 0;
		
		final ClassifierSet parents = new ClassifierSet(null);
		
		indicesToSubsume  = new Vector<Integer>();
		newClassifiersSet = new ClassifierSet(null);
		
		// Select parents
		double fitnessSumLocal = gaSelector.computeFitnessSumNew(evolveSet);
		
		gaSelector.selectWithoutSumNew(1, evolveSet, parents, fitnessSumLocal); 
		final Classifier parentA = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		gaSelector.selectWithoutSumNew(1, evolveSet, parents, fitnessSumLocal);
		final Classifier parentB = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		// Reproduce
		for (int i = 0; i < CHILDREN_PER_GENERATION; i++) {
			
			boolean proceedMyChild = false;
			
			Classifier child;
			// produce a child
			if (Math.random() < crossoverRate && !parentA.equals(parentB)) {
				
				int chromosomeSize = -1;
				if (crossoverOperator == MULTIPOINT_CROSSOVER)
					chromosomeSize = parentA.size() - 2 * (numberOfLabels - 1); 
				else if (crossoverOperator == SINGLEPOINT_CROSSOVER)
					chromosomeSize = parentA.size();
				/*
				 * The point at which the crossover will occur
				 */
				int mutationPoint = (int) Math.round(Math.random() * chromosomeSize - 1);
				child = crossoverOp.operate((i == 0) ? parentB : parentA, (i == 0) ? parentA : parentB, label, mutationPoint);
			} 
			else {
				
				child = (Classifier) ((i == 0) ? parentA : parentB).clone();
				child.setComparisonValue(
						AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION,
						((i == 0) ? parentA : parentB)
								.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
			}

			child = mutationOp.operate(child);
			
			// 0-coverage prevention. every child introduced in the population will be non 0-coverage.
			for (int ins = 0; ins < myLcs.instances.length; ins++) {
				if (child.isMatch(myLcs.instances[ins])) {
					proceedMyChild = true;
					break;
				}
			}

			
			if (proceedMyChild) {
			
				child.inheritParametersFromParents(parentA, parentB);
				myLcs.getClassifierTransformBridge().fixChromosome(child);
				child.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_GA);
				child.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
	
				
				child.created = myLcs.totalRepetition;
				
				long time1 = -System.currentTimeMillis();
				
				int parentIndex = letParentsSubsumeNew(population, parentA, parentB, child);
							
				if ( parentIndex >= 0 )
				{
					indicesToSubsume.add(parentIndex);
				}
				else
				{
					int populationIndex = population.letPopulationSubsume(new Macroclassifier(child, 1), THOROUGHLY_CHECK_WITH_POPULATION);
					
					if ( populationIndex >= 0 )
					{
						indicesToSubsume.add(populationIndex);
					}
					else
					{
						newClassifiersSet.addClassifier( new Macroclassifier(child,1), false);
					}
					
				}
				
				time1 += System.currentTimeMillis();
				
				subsumptionTime += time1; 
				
			}
		}		
		
	}
	
	
	@Override
	public final gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome evolveSetNewSmp (final ClassifierSet evolveSet,
													  final ClassifierSet population,
													  final Random prng, 
													  final int label) {
		
	
		final ClassifierSet parents = new ClassifierSet(null);
		
		final gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome evolutionOutcome = new gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome(); 
		
		Vector<Integer> indicesToSubsumeLocal  = new Vector<Integer>();
		ClassifierSet newClassifiersSetLocal = new ClassifierSet(null);

		double fitnessSum = gaSelector.computeFitnessSumNew(evolveSet);
		
		gaSelector.selectWithoutSumNewSmp(1, evolveSet, parents, fitnessSum, prng); 
		final Classifier parentA = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		gaSelector.selectWithoutSumNewSmp(1, evolveSet, parents, fitnessSum, prng);
		final Classifier parentB = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		long subsumptionTimeLocal = 0;
		
		
		// Reproduce
		for (int i = 0; i < CHILDREN_PER_GENERATION; i++) {
			
			
			Classifier child;
			// produce a child
			if (prng.nextDouble() < crossoverRate && parentA != parentB) {
				
				int chromosomeSize = -1;
				if (crossoverOperator == MULTIPOINT_CROSSOVER)
					chromosomeSize = parentA.size() - 2 * (numberOfLabels - 1); 
				else if (crossoverOperator == SINGLEPOINT_CROSSOVER)
					chromosomeSize = parentA.size();
				/*
				 * The point at which the crossover will occur
				 */
				int mutationPoint = (int) Math.round(prng.nextDouble() * chromosomeSize - 1);
				
				child = crossoverOp.operate((i == 0) ? parentB : parentA, (i == 0) ? parentA : parentB, label, mutationPoint);
			} 
			
			else {
				
				child = (Classifier) ((i == 0) ? parentA : parentB).clone();
				child.setComparisonValue(
						AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION,
						((i == 0) ? parentA : parentB)
								.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
			}

			child = mutationOp.operateSmp(child, prng);
			child.inheritParametersFromParents(parentA, parentB);
			myLcs.getClassifierTransformBridge().fixChromosome(child);
			child.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_GA);
			child.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();

			child.created = myLcs.totalRepetition;
			
			long time1 = -System.currentTimeMillis();
			
			int parentIndex = letParentsSubsumeNew(population, parentA, parentB, child);
						
			if ( parentIndex >= 0 )
			{
				indicesToSubsumeLocal.add(parentIndex);
			}
			else
			{
				int populationIndex = population.letPopulationSubsume(new Macroclassifier(child, 1), THOROUGHLY_CHECK_WITH_POPULATION_SMP);
				
				if ( populationIndex >= 0 )
				{
					indicesToSubsumeLocal.add(populationIndex);
				}
				else
				{
					newClassifiersSetLocal.addClassifier( new Macroclassifier(child,1), false);
				}
				
			}
			
			time1 += System.currentTimeMillis();
			
			subsumptionTimeLocal += time1; 

		}
		
		
		evolutionOutcome.indicesToSubsume = indicesToSubsumeLocal;
		
		evolutionOutcome.newClassifierSet = newClassifiersSetLocal;
		
		evolutionOutcome.subsumptionTime = subsumptionTimeLocal;
				
		return evolutionOutcome;
		
	}
	
	
	@Override
	public final gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome evolveSetNewOneLabelSmp (final ClassifierSet evolveSet,
										   					final ClassifierSet population, 
										   					int label) {
		
		final ClassifierSet parents = new ClassifierSet(null);
		
		final gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome evolutionOutcome = new gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy.EvolutionOutcome(); 
		
		Vector<Integer> indicesToSubsumeLocal  = new Vector<Integer>();
		ClassifierSet newClassifiersSetLocal = new ClassifierSet(null);
		

		
		gaSelector.computeFitnessSum(evolveSet);
		
		gaSelector.selectWithoutSum(1, evolveSet, parents); 
		final Classifier parentA = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		gaSelector.selectWithoutSum(1, evolveSet, parents);
		final Classifier parentB = parents.getClassifier(0);
		parents.deleteClassifier(0);
		
		long subsumptionTimeLocal = 0;
		
		// Reproduce
		for (int i = 0; i < CHILDREN_PER_GENERATION; i++) {
			Classifier child;
			// produce a child
			if ( Math.random() < crossoverRate && parentA != parentB) {
				
				int chromosomeSize = -1;
				if (crossoverOperator == MULTIPOINT_CROSSOVER)
					chromosomeSize = parentA.size() - 2 * (numberOfLabels - 1); 
				else if (crossoverOperator == SINGLEPOINT_CROSSOVER)
					chromosomeSize = parentA.size();
				/*
				 * The point at which the crossover will occur
				 */
				int mutationPoint = (int) Math.round(Math.random() * chromosomeSize - 1);
				child = crossoverOp.operate((i == 0) ? parentB : parentA, (i == 0) ? parentA : parentB, label, mutationPoint);
			} 
			
			else {
				child = (Classifier) ((i == 0) ? parentA : parentB).clone();
				child.setComparisonValue(
						AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION,
						((i == 0) ? parentA : parentB)
								.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
			}

			child = mutationOp.operate(child);
			child.inheritParametersFromParents(parentA, parentB);
			myLcs.getClassifierTransformBridge().fixChromosome(child);
			child.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_GA);
			
			child.created = myLcs.totalRepetition;
			
			long time1 = -System.currentTimeMillis();
			
			int parentIndex = letParentsSubsumeNew(population, parentA, parentB, child);
						
			if ( parentIndex >= 0 )
			{
				indicesToSubsumeLocal.add(parentIndex);
			}
			else
			{
				int populationIndex = population.letPopulationSubsumeNewSmp(new Macroclassifier(child, 1), THOROUGHLY_CHECK_WITH_POPULATION, ptSubsume);
				
				if ( populationIndex >= 0 )
				{
					indicesToSubsumeLocal.add(populationIndex);
				}
				else
				{
					newClassifiersSetLocal.addClassifier( new Macroclassifier(child,1), false);
				}
				
			}
			
			time1 += System.currentTimeMillis();
			
			subsumptionTimeLocal += time1; 
			
		}
		
		evolutionOutcome.indicesToSubsume = indicesToSubsumeLocal;
		evolutionOutcome.newClassifierSet = newClassifiersSetLocal;
		evolutionOutcome.subsumptionTime = subsumptionTimeLocal;
				
		return evolutionOutcome;	
		
	}
	
	@Override
	public final Vector<Integer> getIndicesToSubsume() {
		return indicesToSubsume;
	}
	
	@Override
	public final ClassifierSet getNewClassifiersSet() {
		return newClassifiersSet;
	}
	

	/**
	 * Get the population mean age.
	 * 
	 * @param set
	 *            the set of classifiers to find the mean age
	 * @return an int representing the set's mean age (rounded)
	 */
	public int getMeanAge(final ClassifierSet set) {
		int meanAge = 0;
		// Cache value for optimization
		final int evolveSetSize = set.getNumberOfMacroclassifiers();

		for (int i = 0; i < evolveSetSize; i++) {
			meanAge += set.getClassifierNumerosity(i)
					* set.getClassifier(i).timestamp;
		}
		meanAge /= ((double) set.getTotalNumerosity());

		return meanAge;
	}

	/**
	 * GA Setter.
	 * 
	 * @param age
	 *            the theta_GA
	 */
	public void setThetaGA(int age) {
		this.gaActivationAge = age;
	}

	
	public int getTimestamp() {
		return this.timestamp;
	}
	
	public int getTimestampPerLabel(final int label) {
		return this.timestamps[label];
	}
	
	
	public void increaseTimestamp(){
		timestamp++;
	}
	
	
	
	public int getActivationAge() {
		return this.gaActivationAge;
	}
	
	public long getSubsumptionTime() {
		return subsumptionTime;
	}
	
	public long getSelectionTime() {
		return selectionTime;
	}
	
	public int evolutionConducted() {
		return evolutionConducted;
	}
	
	public int getNumberOfSubsumptionsConducted() {
		return numberOfSubsumptions;
	}
	
	public int getNumberOfNewClassifiers() {
		return numberOfNewClassifiers;
	}
	
	public int getNumberOfDeletionsConducted() {
		return numberOfDeletions;
	}
	
	public long getDeletionTime() {
		return deletionTime;
	}

}