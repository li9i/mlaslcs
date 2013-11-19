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
package gr.auth.ee.lcs.classifiers;

import edu.rit.pj.IntegerForLoop;
import edu.rit.pj.ParallelRegion;
import edu.rit.pj.ParallelSection;
import edu.rit.pj.ParallelTeam;
import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;
import java.text.*;

/**
 * Implement set of Classifiers, counting numerosity for classifiers. This
 * object is serializable.
 * 
 * @author Miltos Allamanis
 * 
 * @has 1 - * Macroclassifier
 * @has 1 - 1 IPopulationControlStrategy
 */
public class ClassifierSet implements Serializable {
	
	
	/* d1 = the number of classifiers whose d is calculated by ns * <F> / f
	 * d2 = the number of classifiers whose d is calculated by ns
	 **/
	public int firstDeletionFormula = 0;
	public int secondDeletionFormula = 0;
	
	public int coveredDeleted = 0;
	public int gaedDeleted = 0;
	
	public int zeroCoverageDeletions = 0;
	public Vector<Integer> zeroCoverageVector = new Vector<Integer>();
	public Vector<Integer> zeroCoverageIterations = new Vector<Integer>();

	
	/**
	 * Serialization id for versioning.
	 */
	private static final long serialVersionUID = 2664983888922912954L;
	
	public int totalGAInvocations = 0;

	public int unmatched;
	
	//public ClassifierSet firstTimeSet;

	
	/**
	 * Open a saved (and serialized) ClassifierSet.
	 * 
	 * @param path
	 *            the path of the ClassifierSet to be opened
	 * @param sizeControlStrategy
	 *            the ClassifierSet's
	 * @param lcs
	 *            the lcs which the new set will belong to
	 * @return the opened classifier set
	 */
	public static ClassifierSet openClassifierSet(final String path,
			final IPopulationControlStrategy sizeControlStrategy,
			final AbstractLearningClassifierSystem lcs) {
		FileInputStream fis = null;
		ObjectInputStream in = null;
		ClassifierSet opened = null;

		try {
			fis = new FileInputStream(path);
			in = new ObjectInputStream(fis);

			opened = (ClassifierSet) in.readObject();
			opened.myISizeControlStrategy = sizeControlStrategy;

			for (int i = 0; i < opened.getNumberOfMacroclassifiers(); i++) {
				final Classifier cl = opened.getClassifier(i);
				cl.setLCS(lcs);
			}

			in.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		} catch (ClassNotFoundException ex) {
			ex.printStackTrace();
		}

		return opened;
	}

	/**
	 * A static function to save the classifier set.
	 * 
	 * @param toSave
	 *            the set to be saved
	 * @param filename
	 *            the path to save the set
	 */
	public static void saveClassifierSet(final ClassifierSet toSave,
			final String filename) {
		FileOutputStream fos = null;
		ObjectOutputStream out = null;

		try {
			fos = new FileOutputStream(filename);
			out = new ObjectOutputStream(fos);
			out.writeObject(toSave);
			out.close();

		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * The total numerosity of all classifiers in set.
	 * @uml.property  name="totalNumerosity"
	 */
	public int totalNumerosity = 0;
	
	public int sumOfUnmatched;
	
	public Vector<Integer> deleteIndices;
	
	public boolean subsumed;
	

	/**
	 * Macroclassifier vector.
	 * @uml.property  name="myMacroclassifiers"
	 * @uml.associationEnd  multiplicity="(0 -1)" elementType="gr.auth.ee.lcs.classifiers.Macroclassifier"
	 */
	private final ArrayList<Macroclassifier> myMacroclassifiers;

	/**
	 * An interface for a strategy on deleting classifiers from the set. This attribute is transient and therefore not serializable.
	 * @uml.property  name="myISizeControlStrategy"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private transient IPopulationControlStrategy myISizeControlStrategy;
	
	static private ClassifierSet matchSetSmp;
	static private Vector<Integer>        deleteIndicesSmp;
	static private ClassifierSet testSetSmp;
	static private int dataInstanceIndexSmp;
	
	static private ClassifierSet matchSetSmp2;
	static private ClassifierSet testSetSmp2;
	static private int dataInstanceIndexSmp2;
	static public Vector<Integer> deleteIndicesSmp2;
	static public ClassifierSet firstTimeSetSmp;	
	static private ArrayList<Integer> candidateDeleteIndicesSmp;
	
	static Vector<Integer> indicesOfSurvivorsSmp;
	static Vector<Float> fitnessOfSurvivorsSmp;
	static Vector<Integer> experienceOfSurvivorsSmp;
	static Vector<Integer> originOfSurvivorsSmp;
		
	static ArrayList<Macroclassifier> myMacroclassifiersSmp;
	
	static Classifier classifierSmp;
	
	static int arrayList = 0;
	
	//padding variables
	long p0,p1,p2,p3,p4,p5,p6,p7;

	/**
	 * The default ClassifierSet constructor.
	 * 
	 * @param sizeControlStrategy
	 *            the size control strategy to use for controlling the set
	 */
	public ClassifierSet(final IPopulationControlStrategy sizeControlStrategy) {
		this.myISizeControlStrategy = sizeControlStrategy;
		this.myMacroclassifiers = new ArrayList<Macroclassifier>();

	}

	/**
	 * Adds a classifier with the a given numerosity to the set. It checks if
	 * the classifier already exists and increases its numerosity. It also
	 * checks for subsumption and updates the set's numerosity.
	 * 
	 * @param thoroughAdd
	 *            to thoroughly check addition
	 * @param macro
	 *            the macroclassifier to add to the set
	 */
	public final void addClassifier(final Macroclassifier macro,
									  final boolean thoroughAdd) {

		final int numerosity = macro.numerosity;
		// Add numerosity to the Set
		this.totalNumerosity += numerosity;
		
		
		subsumed = true;


		// Subsume if possible
		if (thoroughAdd) { // if thoroughAdd = true, before adding the given macro to the population, check it against the whole population for subsumption
			Vector<Integer> indicesVector    = new Vector<Integer>();
			Vector<Float> 	fitnessVector    = new Vector<Float>();
			Vector<Integer> experienceVector = new Vector<Integer>();
			/* 0 gia generality, 1 gia equality */
			Vector<Integer> originVector = new Vector<Integer>();

			
			final Classifier aClassifier = macro.myClassifier;
			for (int i = 0; i < myMacroclassifiers.size(); i++) {
				
				final Classifier theClassifier = myMacroclassifiers.get(i).myClassifier;
				
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						
						indicesVector.add(i);
						originVector.add(0);
						fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
						experienceVector.add(theClassifier.experience);
						
						
						
						/*// Subsume and control size...
						myMacroclassifiers.elementAt(i).numerosity += numerosity;
						myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

						if (myISizeControlStrategy != null) {
							myISizeControlStrategy.controlPopulation(this);
						}
						return;*/
					}
				} else if (theClassifier.equals(aClassifier)) { // Or it can't
																// subsume but
																// it is equal
					indicesVector.add(i);
					originVector.add(1);
					fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
					experienceVector.add(theClassifier.experience);
					
					
					/*myMacroclassifiers.elementAt(i).numerosity += numerosity;
					myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

					if (myISizeControlStrategy != null) {
						myISizeControlStrategy.controlPopulation(this);
					}
					return;*/
				}

			} // kleinei to for gia ton ka9e macroclassifier
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			for (int i = 0; i < indicesVector.size(); i++) {
				if (originVector.elementAt(i) == 0)
						howManyGenerals++;
				else 
					howManyEquals++;
			}
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 0) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 1) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				// Subsume and control size...
				myMacroclassifiers.get(indicesVector.elementAt(indexOfSurvivor)).numerosity += numerosity;
				myMacroclassifiers.get(indicesVector.elementAt(indexOfSurvivor)).numberOfSubsumptions++;
				
				indicesVector.clear();
				originVector.clear();
				fitnessVector.clear();
				experienceVector.clear();
				
				if (myISizeControlStrategy != null) {
					myISizeControlStrategy.controlPopulation(this);
				}
				return;
			}
			
		} // /thoroughadd
		
		subsumed = false;

		/*
		 * No matching or subsumable more general classifier found. Add and
		 * control size...
		 * 
		 * pros9ese ton macroclassifier sto vector myMacroclassifiers.
		 * sti sunexeia an exei oristei stratigiki diagrafis, ektelese tin.
		 * an to numerocity ton macroclassifiers einai pano apo to populationSize arxise na diagrafeis
		 * 
		 * 
		 * an borei na kanei subsume de 9a ektelesei tis parakato entoles (return statements pio pano)
		 */
		//System.out.print(".");
		this.myMacroclassifiers.add(macro);
		if (myISizeControlStrategy != null) {
			myISizeControlStrategy.controlPopulation(this);
		}
	}

	/**
	 * Removes a micro-classifier from the set. It either completely deletes it
	 * (if the classsifier's numerosity is 0) or by decreasing the numerosity.
	 * 
	 * @param aClassifier
	 *            the classifier to delete
	 */
	public final void deleteClassifier(final Classifier aClassifier) {
		
		int index;
		final int macroSize = myMacroclassifiers.size();
		for (index = 0; index < macroSize; index++) {
			if (myMacroclassifiers.get(index).myClassifier.getSerial() ==  aClassifier.getSerial()) {
				break;
			}
		}

		if (index == macroSize)
			return;
		deleteClassifier(index);

	}


public final void addClassifierSmp(final Macroclassifier macro,
			  final boolean thoroughAdd, ParallelTeam ptSubsume ) {
		
		final int numerosity = macro.numerosity;
		// Add numerosity to the Set
		this.totalNumerosity += numerosity;
		

		
		// Subsume if possible
		if (thoroughAdd) { // if thoroughAdd = true, before adding the given macro to the population, check it against the whole population for subsumption
			
			myMacroclassifiersSmp = myMacroclassifiers;	
			classifierSmp = macro.myClassifier;
			
			indicesOfSurvivorsSmp = new Vector<Integer>();
			fitnessOfSurvivorsSmp = new Vector<Float>();
			experienceOfSurvivorsSmp = new Vector<Integer>();
			originOfSurvivorsSmp = new Vector<Integer>();
			
			try {
				ptSubsume.execute(new ParallelRegion() {
				
					public void run() throws Exception
					{
						execute(0,myMacroclassifiersSmp.size()-1,new IntegerForLoop(){
							
							int indexOfSurvivor_thread;
							float fitnessOfSurvivor_thread;
							int experienceOfSurvivor_thread;
							int originOfSurvivor_thread;
														
							long p0,p1,p2,p3,p4,p5,p6,p7;
							long p8,p9,pa,pb,pc,pd,pe,pf;
							
							public void run(int first, int last)
							{
								Vector<Integer> indicesList    = new Vector<Integer>();
								Vector<Float> 	fitnessList    = new Vector<Float>();
								Vector<Integer> experienceList = new Vector<Integer>();
								/* 0 gia generality, 1 gia equality */
								Vector<Integer> originList = new Vector<Integer>();
								
								for (int i = first; i <= last; ++i) {
									
									final Classifier theClassifier = myMacroclassifiersSmp.get(i).myClassifier;
									
									if (theClassifier.canSubsume()) {
										if (theClassifier.isMoreGeneral(classifierSmp)) {
											
											indicesList.add(i);
											originList.add(0);
											fitnessList.add(myMacroclassifiersSmp.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
											experienceList.add(theClassifier.experience);
										}
									} else if (theClassifier.equals(classifierSmp)) { // Or it can't
																					 // subsume but
																					// it is equal
										indicesList.add(i);
										originList.add(1);
										fitnessList.add(myMacroclassifiersSmp.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
										experienceList.add(theClassifier.experience);
									}

								} // kleinei to for gia ton ka9e macroclassifier
								
								int howManyGenerals = 0;
								int howManyEquals = 0;
								for (int i = 0; i < indicesList.size(); i++) {
									if (originList.get(i) == 0)
											howManyGenerals++;
									else 
										howManyEquals++;
								}
								
								int indexOfSurvivor = 0;
								float maxFitness = 0;
								
								if (howManyGenerals !=  0) {
									
									for(int k = 0; k < indicesList.size(); k++) {
										if (originList.get(k) == 0) {
											if (fitnessList.get(k) > maxFitness) {
												maxFitness = fitnessList.get(k);
												indexOfSurvivor = k;
											}
											else if (fitnessList.get(k) == maxFitness) {
												if (experienceList.get(k) >= experienceList.get(indexOfSurvivor)) {
													indexOfSurvivor = k;
												}	
											}
										}
									}
								}
								else if (howManyEquals != 0){
									
									for (int k = 0; k < indicesList.size(); k++) {
										if (originList.get(k) == 1) {
											if (fitnessList.get(k) > maxFitness) {
												maxFitness = fitnessList.get(k);
												indexOfSurvivor = k;
											}
											else if (fitnessList.get(k) == maxFitness) {
												if (experienceList.get(k) >= experienceList.get(indexOfSurvivor)) {
													indexOfSurvivor = k;
												}	
											}
										}
									}
									
								}
								
								indexOfSurvivor_thread = -1;
								
								// if subsumable:
								if (howManyGenerals != 0 || howManyEquals != 0) {
									
									indexOfSurvivor_thread = indicesList.get(indexOfSurvivor);
									originOfSurvivor_thread = originList.get(indexOfSurvivor);
									fitnessOfSurvivor_thread = fitnessList.get(indexOfSurvivor);
									experienceOfSurvivor_thread = experienceList.get(indexOfSurvivor);
									
									indicesList.clear();
									originList.clear();
									fitnessList.clear();
									experienceList.clear();
								}
							}							
							public void finish() throws Exception
							{
								region().critical( new ParallelSection() {
									public void run()
									{
										if ( indexOfSurvivor_thread >= 0 )
										{
											indicesOfSurvivorsSmp.add(indexOfSurvivor_thread);
											originOfSurvivorsSmp.add(originOfSurvivor_thread);
											fitnessOfSurvivorsSmp.add(fitnessOfSurvivor_thread);
											experienceOfSurvivorsSmp.add(experienceOfSurvivor_thread);
										}
									}
								});
							}
							
						});
					}
					

				
				});
			}
			catch( Exception e) {
				e.printStackTrace();
			}
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			if (indicesOfSurvivorsSmp.size() > 0)
			{
				subsumed = true;
				for ( int i = 0 ; i < indicesOfSurvivorsSmp.size(); i++ )
				{
					if (originOfSurvivorsSmp.get(i) == 0)
						howManyGenerals++;
					else
						howManyEquals++;
				}
				
			}
			else
				subsumed = false;
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesOfSurvivorsSmp.size(); k++) {
					if (originOfSurvivorsSmp.get(k) == 0) {
						if (fitnessOfSurvivorsSmp.get(k) > maxFitness) {
							maxFitness = fitnessOfSurvivorsSmp.get(k);
							indexOfSurvivor = k;
						}
						else if (fitnessOfSurvivorsSmp.get(k) == maxFitness) {
							if (experienceOfSurvivorsSmp.get(k) >= experienceOfSurvivorsSmp.get(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesOfSurvivorsSmp.size(); k++) {
					if (originOfSurvivorsSmp.get(k) == 1) {
						if (fitnessOfSurvivorsSmp.get(k) > maxFitness) {
							maxFitness = fitnessOfSurvivorsSmp.get(k);
							indexOfSurvivor = k;
						}
						else if (fitnessOfSurvivorsSmp.get(k) == maxFitness) {
							if (experienceOfSurvivorsSmp.get(k) >= experienceOfSurvivorsSmp.get(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				// Subsume and control size...
				myMacroclassifiers.get(indexOfSurvivor).numerosity += numerosity;
				myMacroclassifiers.get(indexOfSurvivor).numberOfSubsumptions++;
				
				indicesOfSurvivorsSmp.clear();
				originOfSurvivorsSmp.clear();
				fitnessOfSurvivorsSmp.clear();
				experienceOfSurvivorsSmp.clear();
				
				if (myISizeControlStrategy != null) {
					myISizeControlStrategy.controlPopulationSmp(this);
				}
				return;
			}
			
		}
		
		/*
		 * No matching or subsumable more general classifier found. Add and
		 * control size...
		 * 
		 * pros9ese ton macroclassifier sto vector myMacroclassifiers.
		 * sti sunexeia an exei oristei stratigiki diagrafis, ektelese tin.
		 * an to numerocity ton macroclassifiers einai pano apo to populationSize arxise na diagrafeis
		 * 
		 * 
		 * an borei na kanei subsume de 9a ektelesei tis parakato entoles (return statements pio pano)
		 */
		//System.out.print(".");
		this.myMacroclassifiers.add(macro);
		if (myISizeControlStrategy != null) {
			myISizeControlStrategy.controlPopulationSmp(this);
		}
		
	}
	
	/**
	 * Deletes a classifier with the given index. If the macroclassifier at the
	 * given index contains more than one classifier the numerosity is decreased
	 * by one.
	 * 
	 * @param index
	 *            the index of the classifier's macroclassifier to delete
	 */
	public final void deleteClassifier(final int index) {
		
		this.totalNumerosity--; // meiose to numerosity olou tou set
		
		if (this.myMacroclassifiers.get(index).numerosity > 1) {
			this.myMacroclassifiers.get(index).numerosity--; 
		} else {
			//this.myMacroclassifiers.elementAt(index).myClassifier.getLCS().blacklist.addClassifier(new Macroclassifier((this.myMacroclassifiers.elementAt(index).myClassifier), 1), true);
			this.myMacroclassifiers.remove(index); // an to numerosity tou macroclassifier einai 1, diagrapse ton
		}
	}

	
	
	/**
	 * It completely removes the macroclassifier with the given index.
	 * Used by cleanUpZeroCoverage().
	 * 
	 * @param index
	 * 
	 * @author alexandros filotheou
	 * 
	 * */
	
	
	public final void deleteMacroclassifier (final int index) {
		this.totalNumerosity -= this.myMacroclassifiers.get(index).numerosity;
		this.myMacroclassifiers.remove(index);
	
	}
	
	
	
	
	/**
	 * Generate a match set for a given instance.
	 * 
	 * @param dataInstance
	 *            the instance to be matched
	 * @return a ClassifierSet containing the match set
	 */
	public final ClassifierSet generateMatchSet(final double[] dataInstance) {
		final ClassifierSet matchSet = new ClassifierSet(null);
		final int populationSize = this.getNumberOfMacroclassifiers();
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			if (this.getClassifier(i).isMatch(dataInstance)) {
				matchSet.addClassifier(this.getMacroclassifier(i), false);
			}
		}
		return matchSet;
	}

	/**
	 * Generate match set from data instance.
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	public final ClassifierSet generateMatchSet(final int dataInstanceIndex) {
		
		final ClassifierSet matchSet = new ClassifierSet(null); // kataskeuazoume ena adeio arxika classifierSet
		deleteIndices = new Vector <Integer>(); // vector to hold the indices of macroclassifiers that are to be deleted due to zero coverage
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		sumOfUnmatched = 0;

		
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			
			// this = population (macroclassifiers)
			// apo tous macroclassifiers pou sun9etoun ton plh9usmo pairno autous pou einai match me to vision vector
			// getClassifier(i) <--- this.myMacroclassifiers.elementAt(index).myClassifier;
			

			if (this.getClassifier(i).isMatch(dataInstanceIndex)) { 
				
				matchSet.addClassifier(this.getMacroclassifier(i), false); 
			}
			
			sumOfUnmatched += this.getClassifier(i).unmatched;

			
			boolean zeroCoverage = (this.getClassifier(i).getCheckedInstances() >= this.getClassifier(i).getLCS().instances.length) 
									 && (this.getClassifier(i).getCoverage() == 0);
			
			if (this.getClassifier(i).checked == this.getClassifier(i).getLCS().instances.length) 
				this.getClassifier(i).objectiveCoverage = this.getClassifier(i).getCoverage();
			
			if(zeroCoverage) {
				//this.deleteMacroclassifier(i);
				deleteIndices.add(i); // add the index of the macroclassifier with zero coverage 
				System.out.println("deleted due to 0-cov");
			}
		}
		
		for (int i = deleteIndices.size() - 1; i >= 0 ; i--) {
			this.deleteMacroclassifier(deleteIndices.elementAt(i));
		}
		
		//System.out.println(matchSet);

		
		return matchSet;
	}

	
	
	
	
	
	
	/**
	 * Generate match set from data instance. Smp implementation.
	 * 
	 * @author Vag Skar
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	
	public final ClassifierSet generateMatchSetSmp(final int dataInstanceIndex, final ParallelTeam pt){
		
		testSetSmp = this;
		dataInstanceIndexSmp = dataInstanceIndex;
		
		try{
		pt.execute(new ParallelRegion() 
		{
			public void start()
			{
				matchSetSmp = new ClassifierSet(null);
				deleteIndicesSmp = new Vector<Integer>();
			}
			
			public void run() throws Exception
			{
				
				execute(0,testSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop()
				{
					ClassifierSet   regionalMatchSet;
					Vector<Integer> regionalDeleteIndices;
					
					public void start()
					{
						regionalMatchSet      = new ClassifierSet(null);
						regionalDeleteIndices = new Vector<Integer>();
					}
					
					public void run(int first, int last)
					{
						for( int i = first ; i <= last ; ++i)
						{
							if (testSetSmp.getClassifier(i).isMatch(dataInstanceIndexSmp))
							{
								regionalMatchSet.addClassifier(testSetSmp.getMacroclassifier(i),false);
								
								boolean zeroCoverage = (testSetSmp.getClassifier(i).getCheckedInstances() >= testSetSmp.getClassifier(i).getLCS().instances.length) 
										 && (testSetSmp.getClassifier(i).getCoverage() == 0);
								
								if (testSetSmp.getClassifier(i).checked == testSetSmp.getClassifier(i).getLCS().instances.length) 
									testSetSmp.getClassifier(i).objectiveCoverage = testSetSmp.getClassifier(i).getCoverage();
								
								if (zeroCoverage)
									regionalDeleteIndices.add(i);
								
							}
						}
					}
					
					public void finish() throws Exception
					{
						region().critical(new ParallelSection()
						{
							public void run()
							{
								matchSetSmp.merge(regionalMatchSet);
								deleteIndicesSmp.addAll(regionalDeleteIndices);
							}
						});
						regionalMatchSet = null;
						regionalDeleteIndices   = null;
					}
					
				});
			}
			
		});
		}
		catch( Exception e)
		{
			e.printStackTrace();
		}
		
		Collections.sort(deleteIndicesSmp);
		
		
		for (int i = deleteIndicesSmp.size() - 1; i >= 0 ; i--) {
			this.deleteMacroclassifier(deleteIndicesSmp.elementAt(i));
		}
		
		return matchSetSmp;
	}
	
	/**
	 * Generate match set from data instance. New implementation separating the classifiers that match 
	 * the current data instance for the first time. 
	 * 
	 * @author Vag Skar
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	
	public final ClassifierSet generateMatchSetNew(final int dataInstanceIndex){
		
		final ClassifierSet matchSet = new ClassifierSet(null);
		final ClassifierSet firstTimeSet = new ClassifierSet(null);

		deleteIndices = new Vector<Integer>();
		Vector<Integer> candidateDeleteIndices = new Vector<Integer>();
		
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		for ( int i = 0; i < populationSize ; i++ )
		{
			Macroclassifier cl = this.getMacroclassifier(i);
			
			if ( cl.myClassifier.matchInstances == null )
			{
				cl.myClassifier.buildMatches();
			}
			
			
			if ( cl.myClassifier.matchInstances[dataInstanceIndex] == -1 )
			{
				firstTimeSet.addClassifier(cl, false);
				candidateDeleteIndices.add(i);

			}
			else if ( cl.myClassifier.matchInstances[dataInstanceIndex] == 1 )
			{
				matchSet.addClassifier(cl,false);
			}
			
		}
	
		for (int i = 0 ; i < firstTimeSet.getNumberOfMacroclassifiers()  ; i++)
		{
			Macroclassifier cl = firstTimeSet.getMacroclassifier(i);
			cl.myClassifier.matchInstances[dataInstanceIndex]
			= (byte)(cl.myClassifier.getLCS().getClassifierTransformBridge().isMatch
					(cl.myClassifier.getLCS().instances[dataInstanceIndex], cl.myClassifier)? 1 : 0);
			
			cl.myClassifier.checked++;
			cl.myClassifier.covered += cl.myClassifier.matchInstances[dataInstanceIndex];
			
			if(cl.myClassifier.matchInstances[dataInstanceIndex] == 1)
				matchSet.addClassifier(cl, false);
			
			boolean zeroCoverage = (cl.myClassifier.checked >= cl.myClassifier.getLCS().instances.length) 
			                       && (cl.myClassifier.covered == 0);
			
			if (cl.myClassifier.checked == cl.myClassifier.getLCS().instances.length) 
				cl.myClassifier.objectiveCoverage = cl.myClassifier.getCoverage();
			
			if (zeroCoverage)
				deleteIndices.add(candidateDeleteIndices.get(i));			
		}	
		
		
		for ( int i = deleteIndices.size() - 1 ; i >= 0 ; i-- )
		{
			zeroCoverageIterations.add(myMacroclassifiers.get(deleteIndices.elementAt(i)).myClassifier.getLCS().totalRepetition);

			this.deleteMacroclassifier(deleteIndices.elementAt(i));
			zeroCoverageDeletions++;
			zeroCoverageVector.add(zeroCoverageDeletions);
			//System.out.println("deleted due to 0-cov");
		}

		firstTimeSetSmp = firstTimeSet;
		
		deleteIndices.clear();
		candidateDeleteIndices.clear();
		
		return matchSet;
	}
	
	/**
	 * Generate match set from data instance. New implementation separating the classifiers that match 
	 * the current data instance for the first time. Smp implementation.
	 * 
	 * @author Vag Skar
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	
	
	public final ClassifierSet generateMatchSetNewSmp(int dataInstanceIndex,ParallelTeam pt){
		
		testSetSmp2 = this;
		dataInstanceIndexSmp2 = dataInstanceIndex;
		
		try{
			pt.execute(new ParallelRegion(){
			
				public void start()
				{
					firstTimeSetSmp = new ClassifierSet(null);
					matchSetSmp2    = new ClassifierSet(null);
					candidateDeleteIndicesSmp = new ArrayList<Integer>();
					deleteIndicesSmp2 = new Vector<Integer>();
				}
			
				public void run() throws Exception
				{
					execute(0,testSetSmp2.getNumberOfMacroclassifiers()-1,new IntegerForLoop(){
						
						ClassifierSet matchSet_thread;
						ClassifierSet firstTimeSet_thread;
						ArrayList<Integer> candidateDeleteIndices_thread = new ArrayList<Integer>();
												
						//padding variables
						long p0,p1,p2,p3,p4,p5,p6,p7;
						
						public void start()
						{
							matchSet_thread = new ClassifierSet(null);
							firstTimeSet_thread = new ClassifierSet(null);
						}
						
						public void run(int first,int last)
						{
							for ( int i = first ; i <= last ; ++i )
							{
								Macroclassifier cl = testSetSmp2.getMacroclassifier(i);
								if ( cl.myClassifier.matchInstances == null )
								{
									cl.myClassifier.buildMatches();
								}
								
								if( cl.myClassifier.matchInstances[dataInstanceIndexSmp2] == -1 )
								{
									firstTimeSet_thread.addClassifier(cl, false);
									candidateDeleteIndices_thread.add(i);
								}
								else if ( cl.myClassifier.matchInstances[dataInstanceIndexSmp2] == 1 )
								{
									matchSet_thread.addClassifier(cl,false);
								}
								
							}	
						}
						
						public void finish() throws Exception
						{
							region().critical(new ParallelSection(){
								
								public void run()
								{
									firstTimeSetSmp.merge(firstTimeSet_thread);
									matchSetSmp2.merge(matchSet_thread);
									candidateDeleteIndicesSmp.addAll(candidateDeleteIndices_thread);
								}
								
							});
						}
						
						
					});
					
					execute(0,firstTimeSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop(){
						
						Vector<Integer> deleteIndices_thread;
						ClassifierSet matchSet_thread;
						
						//padding variables
						long p0,p1,p2,p3,p4,p5,p6,p7;
						
						public void start()
						{
							deleteIndices_thread = new Vector<Integer>();
							matchSet_thread = new ClassifierSet(null);
						}
						
						public void run(int first,int last)
						{
							for (int i = first ; i <= last  ; ++i)
							{
								Macroclassifier cl = firstTimeSetSmp.getMacroclassifier(i);
								cl.myClassifier.matchInstances[dataInstanceIndexSmp2]
								= (byte)(cl.myClassifier.getLCS().getClassifierTransformBridge().isMatch
										(cl.myClassifier.getLCS().instances[dataInstanceIndexSmp2], cl.myClassifier)? 1 : 0);
								cl.myClassifier.checked++;
								cl.myClassifier.covered += cl.myClassifier.matchInstances[dataInstanceIndexSmp2];
								
								if(cl.myClassifier.matchInstances[dataInstanceIndexSmp2] == 1)
									matchSet_thread.addClassifier(cl, false);
								
								boolean zeroCoverage = (cl.myClassifier.checked >= cl.myClassifier.getLCS().instances.length) 
								                       && (cl.myClassifier.covered == 0);
								
								if (cl.myClassifier.checked == cl.myClassifier.getLCS().instances.length) 
									cl.myClassifier.objectiveCoverage = cl.myClassifier.getCoverage();
								
								if (zeroCoverage)
									deleteIndices_thread.add(candidateDeleteIndicesSmp.get(i));
								
							}							
						}
						
						public void finish() throws Exception
						{
							region().critical( new ParallelSection() {								
								public void run()
								{
									deleteIndicesSmp2.addAll(deleteIndices_thread);
									matchSetSmp2.merge(matchSet_thread);
								}
								
							});

						}						
						
					});
					
				}			
			
			});
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
		Collections.sort(deleteIndicesSmp2);
		
		for (int i = deleteIndicesSmp2.size()-1; i >=0 ; i--)
		{
			this.deleteMacroclassifier(deleteIndicesSmp2.elementAt(i));
		}
		
		return matchSetSmp2;
		
	}
	
	
	
	
	
	public final ClassifierSet generateMatchSetCached (final int dataInstanceIndex) {
		
		final ClassifierSet matchSet = new ClassifierSet(null);
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		for (int i= 0; i < populationSize; i++) {
			if (this.getClassifier(i).isMatchCached(dataInstanceIndex)) {
				matchSet.addClassifier(this.getMacroclassifier(i), false);
			}
		}
		return matchSet;
	}
	
	
	
	
	
	
	
	/**
	 * Return the classifier at a given index of the macroclassifier vector.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the classifier at the specified index
	 */
	public final Classifier getClassifier(final int index) {
		return this.myMacroclassifiers.get(index).myClassifier;
	}

	/**
	 * Returns a classifier's numerosity (the number of microclassifiers).
	 * 
	 * @param aClassifier
	 *            the classifier
	 * @return the given classifier's numerosity
	 */
	public final int getClassifierNumerosity(final Classifier aClassifier) {
		for (int i = 0; i < myMacroclassifiers.size(); i++) {
			if (myMacroclassifiers.get(i).myClassifier.getSerial() == aClassifier
					.getSerial()) // ka9e (micro)classifier exei kai ena serial number. 
								  // apo oti exo katalabei diaforetiko gia ka9e microclassifier, 
								  // akoma kai gia autous tou idiou macroclassifier
				//System.out.println("myClassifier" + myMacroclassifiers.elementAt(i).myClassifier.getSerial());
				//System.out.println(aClassifier.getSerial());
				return this.myMacroclassifiers.get(i).numerosity;
		}
		return 0;
	}

	/**
	 * Overloaded function for getting a numerosity.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the index'th macroclassifier numerosity
	 */
	public final int getClassifierNumerosity(final int index) {
		return this.myMacroclassifiers.get(index).numerosity;
	}

	/**
	 * Returns (a copy of) the macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 */
	public final Macroclassifier getMacroclassifier(final int index) {
		return new Macroclassifier(this.myMacroclassifiers.get(index));
		//return this.myMacroclassifiers.elementAt(index);
	}
	
	
	
	/**
	 * returns the myMacroclassifiers vector
	 * 
	 * @author alexandros filotheou
	 * 
	 */
	public ArrayList<Macroclassifier> getMacroclassifiersVector() {
		return myMacroclassifiers;
	}
	
	/**
	 * Returns the actual (not a copy as the aboce method) macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 * 
	 * @author alexandros filotheou
	 */
	
	public Macroclassifier getActualMacroclassifier(final int index) {
		return this.myMacroclassifiers.get(index);
	}
	
	/**
	 * Returns the actual (not a copy as the aboce method) macroclassifier that corresponds to the aClassifier classifier.
	 * 
	 * @param aClassifier
	 *            the classifier whose corresponding Macroclassifier we wish to obtain
	 * @return the macroclassifier
	 * 
	 * @author alexandros filotheou
	 */
		
	
	public Macroclassifier getActualMacroclassifier(final Classifier aClassifier) {
		
		for (int i = 0; i < myMacroclassifiers.size(); i++) {
			if (myMacroclassifiers.get(i).myClassifier.getSerial() == aClassifier.getSerial()) 
				return this.myMacroclassifiers.get(i);
		}
		return null;	
	}


	/**
	 * Getter.
	 * 
	 * @return the number of macroclassifiers in the set
	 */
	public final int getNumberOfMacroclassifiers() {
		return this.myMacroclassifiers.size();
	}

	/**
	 * Get the set's population control strategy
	 * 
	 * @return the set's population control strategy
	 */
	public final IPopulationControlStrategy getPopulationControlStrategy() {
		return myISizeControlStrategy;
	}

	/**
	 * Returns the set's total numerosity (the total number of microclassifiers).
	 * @return  the sets total numerosity
	 * @uml.property  name="totalNumerosity"
	 */
	public final int getTotalNumerosity() {
		return this.totalNumerosity;
	}

	/**
	 * @return true if the set is empty
	 */
	public final boolean isEmpty() {
		return this.myMacroclassifiers.isEmpty();
	}
	
	
	
	
	public final int letPopulationSubsume(final Macroclassifier macro,
			  final boolean thoroughAdd) {
		
		// Subsume if possible
		if (thoroughAdd) { // if thoroughAdd = true, before adding the given macro to the population, check it against the whole population for subsumption
			Vector<Integer> indicesVector    = new Vector<Integer>();
			Vector<Float> 	fitnessVector    = new Vector<Float>();
			Vector<Integer> experienceVector = new Vector<Integer>();
			/* 0 gia generality, 1 gia equality */
			Vector<Integer> originVector = new Vector<Integer>();

			
			final Classifier aClassifier = macro.myClassifier;
			for (int i = 0; i < myMacroclassifiers.size(); i++) {
				
				final Classifier theClassifier = myMacroclassifiers.get(i).myClassifier;
				
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						
						indicesVector.add(i);
						originVector.add(0);
						fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
						experienceVector.add(theClassifier.experience);
						
						
						
						/*// Subsume and control size...
						myMacroclassifiers.elementAt(i).numerosity += numerosity;
						myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

						if (myISizeControlStrategy != null) {
							myISizeControlStrategy.controlPopulation(this);
						}
						return;*/
					}
				} else if (theClassifier.equals(aClassifier)) { // Or it can't
																// subsume but
																// it is equal
					indicesVector.add(i);
					originVector.add(1);
					fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
					experienceVector.add(theClassifier.experience);
					
					
					/*myMacroclassifiers.elementAt(i).numerosity += numerosity;
					myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

					if (myISizeControlStrategy != null) {
						myISizeControlStrategy.controlPopulation(this);
					}
					return;*/
				}

			} // kleinei to for gia ton ka9e macroclassifier
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			for (int i = 0; i < indicesVector.size(); i++) {
				if (originVector.elementAt(i) == 0)
						howManyGenerals++;
				else 
					howManyEquals++;
			}
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 0) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 1) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				
				int toBeReturned = indicesVector.elementAt(indexOfSurvivor);
				
				indicesVector.clear();
				originVector.clear();
				fitnessVector.clear();
				experienceVector.clear();
				
				return toBeReturned;
			}
			
		} // /thoroughadd
		
		return -1;
		
	}

	
	
	
	
		public final int letPopulationSubsumeNewSmp (final Macroclassifier macro,
					  									final boolean thoroughAdd,
					  									final ParallelTeam ptSubsume) {		

		// Subsume if possible
		if (thoroughAdd) { // if thoroughAdd = true, before adding the given macro to the population, check it against the whole population for subsumption
			
			myMacroclassifiersSmp = myMacroclassifiers;	
			classifierSmp = macro.myClassifier;
			
			indicesOfSurvivorsSmp = new Vector<Integer>();
			fitnessOfSurvivorsSmp = new Vector<Float>();
			experienceOfSurvivorsSmp = new Vector<Integer>();
			originOfSurvivorsSmp = new Vector<Integer>();
			
			try {
				ptSubsume.execute(new ParallelRegion() {
				
					public void run() throws Exception
					{
						execute(0,myMacroclassifiersSmp.size()-1,new IntegerForLoop(){
							
							int indexOfSurvivor_thread;
							float fitnessOfSurvivor_thread;
							int experienceOfSurvivor_thread;
							int originOfSurvivor_thread;
														
							long p0,p1,p2,p3,p4,p5,p6,p7;
							long p8,p9,pa,pb,pc,pd,pe,pf;
							
							public void run(int first, int last)
							{
								int threadIndex = getThreadIndex();
								Vector<Integer> indicesVector    = new Vector<Integer>();
								Vector<Float> 	fitnessVector    = new Vector<Float>();
								Vector<Integer> experienceVector = new Vector<Integer>();
								/* 0 gia generality, 1 gia equality */
								Vector<Integer> originVector = new Vector<Integer>();
								
								for (int i = first; i <= last; ++i) {
									
									final Classifier theClassifier = myMacroclassifiersSmp.get(i).myClassifier;
									
									if (theClassifier.canSubsume()) {
										if (theClassifier.isMoreGeneral(classifierSmp)) {
											
											indicesVector.add(i);
											originVector.add(0);
											fitnessVector.add(myMacroclassifiersSmp.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
											experienceVector.add(theClassifier.experience);
										}
									} else if (theClassifier.equals(classifierSmp)) { // Or it can't
																					 // subsume but
																					// it is equal
										indicesVector.add(i);
										originVector.add(1);
										fitnessVector.add(myMacroclassifiersSmp.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
										experienceVector.add(theClassifier.experience);
									}

								} // kleinei to for gia ton ka9e macroclassifier
								
								int howManyGenerals = 0;
								int howManyEquals = 0;
								for (int i = 0; i < indicesVector.size(); i++) {
									if (originVector.get(i) == 0)
											howManyGenerals++;
									else 
										howManyEquals++;
								}
								
								int indexOfSurvivor = 0;
								float maxFitness = 0;
								
								if (howManyGenerals !=  0) {
									
									for(int k = 0; k < indicesVector.size(); k++) {
										if (originVector.get(k) == 0) {
											if (fitnessVector.get(k) > maxFitness) {
												maxFitness = fitnessVector.get(k);
												indexOfSurvivor = k;
											}
											else if (fitnessVector.get(k) == maxFitness) {
												if (experienceVector.get(k) >= experienceVector.get(indexOfSurvivor)) {
													indexOfSurvivor = k;
												}	
											}
										}
									}
								}
								else if (howManyEquals != 0){
									
									for (int k = 0; k < indicesVector.size(); k++) {
										if (originVector.get(k) == 1) {
											if (fitnessVector.get(k) > maxFitness) {
												maxFitness = fitnessVector.get(k);
												indexOfSurvivor = k;
											}
											else if (fitnessVector.get(k) == maxFitness) {
												if (experienceVector.get(k) >= experienceVector.get(indexOfSurvivor)) {
													indexOfSurvivor = k;
												}	
											}
										}
									}
									
								}
								
								indexOfSurvivor_thread = -1;
								
								// if subsumable:
								if (howManyGenerals != 0 || howManyEquals != 0) {
									
									indexOfSurvivor_thread = indicesVector.get(indexOfSurvivor);
									originOfSurvivor_thread = originVector.get(indexOfSurvivor);
									fitnessOfSurvivor_thread = fitnessVector.get(indexOfSurvivor);
									experienceOfSurvivor_thread = experienceVector.get(indexOfSurvivor);
									
									indicesVector.clear();
									originVector.clear();
									fitnessVector.clear();
									experienceVector.clear();
								}
							}							
							public void finish() throws Exception
							{
								region().critical( new ParallelSection() {
									public void run()
									{
										if ( indexOfSurvivor_thread >= 0 )
										{
											indicesOfSurvivorsSmp.add(indexOfSurvivor_thread);
											originOfSurvivorsSmp.add(originOfSurvivor_thread);
											fitnessOfSurvivorsSmp.add(fitnessOfSurvivor_thread);
											experienceOfSurvivorsSmp.add(experienceOfSurvivor_thread);
										}
									}
								});
							}
							
						});
					}
					

				
				});
			}
			catch( Exception e) {
				e.printStackTrace();
			}
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			if (indicesOfSurvivorsSmp.size() > 0)
			{
				subsumed = true;
				for ( int i = 0 ; i < indicesOfSurvivorsSmp.size(); i++ )
				{
					if (originOfSurvivorsSmp.get(i) == 0)
						howManyGenerals++;
					else
						howManyEquals++;
				}
				
			}
			else
				subsumed = false;
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesOfSurvivorsSmp.size(); k++) {
					if (originOfSurvivorsSmp.get(k) == 0) {
						if (fitnessOfSurvivorsSmp.get(k) > maxFitness) {
							maxFitness = fitnessOfSurvivorsSmp.get(k);
							indexOfSurvivor = k;
						}
						else if (fitnessOfSurvivorsSmp.get(k) == maxFitness) {
							if (experienceOfSurvivorsSmp.get(k) >= experienceOfSurvivorsSmp.get(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesOfSurvivorsSmp.size(); k++) {
					if (originOfSurvivorsSmp.get(k) == 1) {
						if (fitnessOfSurvivorsSmp.get(k) > maxFitness) {
							maxFitness = fitnessOfSurvivorsSmp.get(k);
							indexOfSurvivor = k;
						}
						else if (fitnessOfSurvivorsSmp.get(k) == maxFitness) {
							if (experienceOfSurvivorsSmp.get(k) >= experienceOfSurvivorsSmp.get(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				// Subsume and control size...
								
				indicesOfSurvivorsSmp.clear();
				originOfSurvivorsSmp.clear();
				fitnessOfSurvivorsSmp.clear();
				experienceOfSurvivorsSmp.clear();
				
				return indexOfSurvivor;
			}			
			
		}
		
		return -1;
		
	}
	
	
	

	/**
	 * Merge a set into this set.
	 * 
	 * @param aSet
	 *            the set to be merged.
	 */
	public final void merge(final ClassifierSet aSet) {
		final int setSize = aSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			final Macroclassifier ml = aSet.getMacroclassifier(i);
			this.addClassifier(ml, false);
		}
	}
	
	
	
	public final void mergeWithoutControl(final ClassifierSet aSet) {
		final int setSize = aSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			final Macroclassifier ml = aSet.getMacroclassifier(i);
			final int numerosity = ml.numerosity;
			this.totalNumerosity += numerosity;
			this.myMacroclassifiers.add(ml);			
		}
	}

	

	/**
	 * Print all classifiers in the set.
	 */
	public final void print() {
		System.out.println(toString());
	}

	/**
	 * Remove all set's macroclassifiers.
	 */
	public final void removeAllMacroclassifiers() {
		this.myMacroclassifiers.clear();
		this.totalNumerosity = 0;
	}

	/**
	 * Self subsume. the fuck?
	 */
	public final void selfSubsume() {
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			final Macroclassifier cl = this.getMacroclassifier(0);
			final int numerosity = cl.numerosity;
			this.myMacroclassifiers.remove(0);
			this.totalNumerosity -= numerosity;
			this.addClassifier(cl, true);
		}
	}

	
	@Override
	public String toString() { // o buffer grafei sto arxeio population.txt. ta system.out stin konsola
		final StringBuffer response = new StringBuffer();
		
		double numOfCover = 0;
		double numOfGA = 0;
		double numOfInit = 0;
		int 	numOfSubsumptions = 0;
		int 	meanNs = 0;
		int 	coveredTotalNumerosity = 0;
		int 	gaedTotalNumerosity = 0;
		double meanAcc = 0;
		
		//int numberOfFinalClassifiersNotSeenTheWholePicture = 0;
		//int numInstances = this.getClassifier(0).getLCS().instances.length;
		
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
/*			myMacroclassifiers.elementAt(i).totalFitness = 
				this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity;*/
			double acc = this.getActualMacroclassifier(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
			if (Double.isNaN(acc)) 
				acc = 0;
			
			meanNs += this.getClassifier(i).getNs();
			meanAcc += acc * this.getMacroclassifier(i).numerosity;
/*			if (this.getClassifier(i).getCheckedInstances() < numInstances)
				numberOfFinalClassifiersNotSeenTheWholePicture++;*/
		}
		
/*		if (numberOfFinalClassifiersNotSeenTheWholePicture > 0)
			System.out.println(numberOfFinalClassifiersNotSeenTheWholePicture + " rules not seen the entire dataset even once");*/
		
		if (this.getNumberOfMacroclassifiers() > 0) {
			meanNs /= this.getNumberOfMacroclassifiers();
			meanAcc /= this.getTotalNumerosity();
		}
		
        DecimalFormat df = new DecimalFormat("#.####");

        
        double accuracyOfCovered = 0;
        double accuracyOfGa = 0;
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {

			
			//response.append(this.getClassifier(i).toString()
			response.append(
					myMacroclassifiers.get(i).myClassifier.toString() // antecedent => concequent
					+ "|"	
					//+ " total fitness: " + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity
					// myMacroclassifiers.elementAt(i).toString isos kalutera
					+ "macro fit:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION) 
							* myMacroclassifiers.get(i).numerosity) 
					+ "|"
					+ "fit:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION))
					+ "|"
					+ "acc:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY))
					+ "|"
					+ "num:|" + myMacroclassifiers.get(i).numerosity 
					+ "|"
					+ "exp:|" + myMacroclassifiers.get(i).myClassifier.experience  
					+ "|"
					+ "cov:|" + (int) (myMacroclassifiers.get(i).myClassifier.objectiveCoverage * myMacroclassifiers.get(i).myClassifier.getLCS().instances.length)
					+ "|");
			
			response.append(myMacroclassifiers.get(i).myClassifier.getUpdateSpecificData());
			
			//response.append(" deleted by: " + myMacroclassifiers.elementAt(i).myClassifier.formulaForD);
			
			if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER) {
				numOfCover++;
				coveredTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfCovered +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|cover" + "|");
			}
			else if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
				numOfGA++;
				gaedTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfGa +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|ga" + "|");
			}
			else if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
				numOfInit++;
				coveredTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfCovered +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|init "+ "|");
			}	
			
			
			numOfSubsumptions += myMacroclassifiers.get(i).numberOfSubsumptions;
			//response.append(" created: " + myMacroclassifiers.elementAt(i).myClassifier.created + " ");
			response.append("created:|" + myMacroclassifiers.get(i).myClassifier.cummulativeInstanceCreated + "|");
			response.append("last in correctset:|" + myMacroclassifiers.get(i).myClassifier.timestamp + "|");
			response.append("subsumptions:|" + myMacroclassifiers.get(i).numberOfSubsumptions + "|");
			response.append("created:|" + (-Integer.MIN_VALUE + myMacroclassifiers.get(i).myClassifier.getSerial()) + "th" + "|");
			response.append(System.getProperty("line.separator"));
		}
		
		System.out.println("\nPopulation size (macro, micro): "  	+ "(" + this.getNumberOfMacroclassifiers() + "," + this.getTotalNumerosity() + ")");

		System.out.println("Classifiers in population covered: " 	+ (int) numOfCover);
		System.out.println("Classifiers in population ga-ed:   " 	+ (int) numOfGA);
		System.out.println("Classifiers in population init-ed: " 	+ (int) numOfInit);
		System.out.println();
		
		System.out.println("Accuracy of covered: " +  (Double.isNaN(accuracyOfCovered / numOfCover) ? 0 : accuracyOfCovered / coveredTotalNumerosity /*(numOfCover + numOfInit)*/));
		System.out.println("Accuracy of gaed:    " +  (Double.isNaN(accuracyOfGa/ numOfCover) ? 0 : accuracyOfGa / gaedTotalNumerosity/*numOfGA*/));
		System.out.println();

		
		System.out.println("Mean ns:   " + meanNs);
		System.out.println("Mean pure accuracy:   " + meanAcc);
		
		System.out.println("ga invocations: " 						+ this.totalGAInvocations);

		System.out.println("Subsumptions: " + numOfSubsumptions + "\n");
		//System.out.println("Total number of epochs:" + this.getClassifier(this.getNumberOfMacroclassifiers() - 1).timestamp);


//		String foo = null;
//		return foo;
		return response.toString();
	}

}