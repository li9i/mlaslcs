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

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Represents a single classifier/ rule. Connects to the representation through
 * a visitor pattern.
 * 
 * @author Miltos Allamanis
 */
public final class Classifier extends ExtendedBitSet implements Serializable {

	/**
	 * Static method to create Classifiers.
	 * 
	 * @param lcs
	 *            the LCS instance being used
	 * @return the new classifier
	 */
	public static Classifier createNewClassifier(
			final AbstractLearningClassifierSystem lcs) {
		return new Classifier(lcs);
	}

	/**
	 * Static method to create Classifiers.
	 * 
	 * @param lcs
	 *            the lcs being used
	 * @param chromosome
	 *            the chromosome to be copied
	 * @return the new classifier
	 */
	public static Classifier createNewClassifier(
			final AbstractLearningClassifierSystem lcs,
			final ExtendedBitSet chromosome) {
		return new Classifier(lcs, chromosome);
	}

	/**
	 * The transform bridge.
	 * @uml.property  name="transformBridge"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private transient ClassifierTransformBridge transformBridge;

	/**
	 * Update Strategy.
	 * @uml.property  name="updateStrategy"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private transient AbstractUpdateStrategy updateStrategy;

	/**
	 * The LCS instance.
	 * @uml.property  name="myLcs"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private transient AbstractLearningClassifierSystem myLcs;

	/**
	 * The initial Fitness of a classifier.
	 */
	private static final double INITIAL_FITNESS = 0.5;

	/**
	 * Cache for action.
	 * @uml.property  name="actionCache" multiplicity="(0 -1)" dimension="1"
	 */
	private int[] actionCache = null;

	/**
	 * Serialization code for versioning.
	 */
	private static final long serialVersionUID = 8628765535406768159L;

	/**
	 * An object (of undefined type) that is used by the update algorithms.
	 * @uml.property  name="updateData"
	 */
	private Serializable updateData;
	
	private Serializable[] updateDataArray;

	/**
	 * A boolean array indicating which dataset instances the rule matches.
	 * @uml.property  name="matchInstances" multiplicity="(0 -1)" dimension="1"
	 */
	public transient byte[] matchInstances; // logika, byte, giati pairnei times -1 , 1 kai krata ligotero xoro apo int
											 // alla den borei na ton kanei boolean. 
											 // 9a borouse na ginei boolean omos! tsekare isMatch(int)

	/**
	 * A float showing the number of instances that the rule has covered. Used for calculating coverage.
	 * @uml.property  name="covered"
	 */
	public transient int covered = 0;

	/**
	 * The number of instances we have checked so far. Used for coverage
	 * @uml.property  name="checked"
	 */
	public transient int checked = 0;

	/**
	 * The serial number of last classifier (start from the lowest & increment).
	 */
	private static int currentSerial = Integer.MIN_VALUE;

	/**
	 * The serial number of the classifier.
	 * @uml.property  name="serial"
	 */
	private int serial;

	/**
	 * The classifier's experience.
	 * @uml.property  name="experience"
	 */
	public int experience = 0;

	/**
	 * The timestamp is the last iteration the classifier has participated in a GA Evolution.
	 * @uml.property  name="timestamp"
	 */
	public int timestamp = 0;
	
	
	private final int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels",1);

	public int timestamps [] = new int [numberOfLabels];

	/**
	 * A boolean representing the classifier's ability to subsume.
	 * @uml.property  name="subsumes"
	 */
	private boolean subsumes = false;

	/**
	 * An object for saving the transformation specific data. sti 9esi 0 to action
	 * @uml.property  name="transformData"
	 */
	public Serializable transformData;
	
	private int origin;
	
	public static final int CLASSIFIER_ORIGIN_INIT = 0;
	
	public static final int CLASSIFIER_ORIGIN_COVER = 1;
	
	public static final int CLASSIFIER_ORIGIN_GA = 2;

	
	public int created;
	
	public int cummulativeInstanceCreated = 0;
	
	
	
	/*
	 * ka9e kanonas gennietai me objectiveCoverage == -1. otan dei gia proti fora ola ta deigmata tou dataset (meso tis generate match set)
	 * tote i metabliti pairnei tin pragmatiki, antikeimeniki, timi tou covered / checked.
	 * 
	 * uparxei periptosi oi kanones pou diagrafontai na min exoun dei oute mia fora to dataset kai omos na diagrafontai
	 * */
	public double objectiveCoverage = -1;
	
	public int formulaForD = -1;
	
	public int unmatched;
	
	/**
	 * The default constructor. Creates a chromosome of the given size
	 * 
	 * @param lcs
	 *            the LCS instance that the new classifier will belong to
	 */
	private Classifier(final AbstractLearningClassifierSystem lcs) {
		
		super(lcs.getClassifierTransformBridge().getChromosomeSize()); // dimiourgei to xromosoma. arxika einai olo midenika.
		this.transformBridge = lcs.getClassifierTransformBridge();
		this.updateStrategy = lcs.getUpdateStrategy();
		myLcs = lcs;
		setConstructionData(); // arxikopoiei tis metablites epidosis tou classifier, 
							   // to serial tou gia na ton anagnorizoume kai to representation tou
	}

	/**
	 * Constructor for creating a classifier from a chromosome.
	 * 
	 * 
	 * @param chromosome
	 *            the chromosome from which to create the classifier
	 * @param lcs
	 *            the LCS instance that the classifier belongs to
	 */
	private Classifier(final AbstractLearningClassifierSystem lcs,
			final ExtendedBitSet chromosome) {
		super(chromosome); // Akribos idia me tin parapano, me ti diafora oti dino ego to xromosoma 
						   // gia na to kanei copy, anti na kataskeuastei ena kainourio midenismeno
		this.transformBridge = lcs.getClassifierTransformBridge();
		this.updateStrategy = lcs.getUpdateStrategy();
		myLcs = lcs;
		setConstructionData();
	}
	
	/**
	 * @param strategy
	 * @uml.property  name="updateStrategy"
	 */
	public final void setUpdateStrategy(AbstractUpdateStrategy strategy) {
		this.updateStrategy = strategy;
	}
	
	public final void setUpdateObject(Serializable obj) {
		this.updateData = obj;
	}

	/**
	 * Build matches vector (with train instances) and initialize it.
	 */
	public void buildMatches() {
		this.matchInstances = new byte[myLcs.instances.length];
		Arrays.fill(this.matchInstances, (byte) -1); // gemise to me -1
	}

	/**
	 * Getter for the subsumption ability.
	 * 
	 * @return true if the classifier is strong enough to subsume
	 */
	public boolean canSubsume() {
		return subsumes;
	}

	/**
	 * A representation specific method representing the classifier's ability to
	 * correctly classify a train instance.
	 * 
	 * @param instanceIndex
	 *            the index of the train instance
	 * @return a number that represents the correctness. This number may be 0,1
	 *         for unilabel classification but it may also be in the range [0,1]
	 */
	public float classifyCorrectly(final int instanceIndex) {
		return transformBridge.classifyAbilityAll(this, instanceIndex);
	}

	/**
	 * Returns the classification ability of classifier for a specific label.
	 * 
	 * @param instanceIndex
	 *            the instance of the index
	 * @param labelIndex
	 *            the index of the label
	 * @return a float representing the classifier's ability to classify for
	 *         this instance
	 */
	public float classifyLabelCorrectly(final int instanceIndex,
										 final int labelIndex) {
		return transformBridge.classifyAbilityLabel(this, instanceIndex,
				labelIndex);
	}

	/**
	 * Clone of the classifier.
	 * 
	 * @return the clone
	 */
	@Override
	public Object clone() {
		return new Classifier(myLcs, this);
	}

	/**
	 * @param anotherClassifier
	 *            the classifier against which check for equality
	 * @return true if the classifiers have equal chromosomes
	 */
	public boolean equals(final Classifier anotherClassifier) {
		return transformBridge.areEqual(this, anotherClassifier);
	}

	/**
	 * Calls the bridge to fix itself.
	 */
	public void fixChromosome() {
		transformBridge.fixChromosome(this);
	}

	/**
	 * Getter for the advocated action.
	 * 
	 * @return the advocated action
	 */
	public int[] getActionAdvocated() {
		if (actionCache == null) {
			actionCache = transformBridge.getClassification(this);
		}
		return actionCache;
	}

	/**
	 * Getter of the number of instances the classifier has seen.
	 * 
	 * @return the number of instances the classifier has checked
	 */
	public int getCheckedInstances() {
		return checked;
	}

	public int getCoveredInstances() {
		return covered;
	}
	
	public AbstractLearningClassifierSystem getLCS() {
		return myLcs;
	}
	

	
	public int getClassifierOrigin(){
		return this.origin;
	}
	
/*	public void setDateCreated(int dateCreated) {
		this.timestamp = dateCreated;
	}
	
	public int getDateCreated() {
		return timestamp;
	}*/
	/**
	 * Returns a numeric value for comparing the classifier.
	 * 
	 * @param mode
	 *            the mode of comparison
	 * @return the value of comparison
	 */
	public double getComparisonValue(final int mode) {
		return updateStrategy.getComparisonValue(this, mode);
	}

	/**
	 * Returns the classifer's coverage approximation.
	 * 
	 * @return the classifier's coverage as calculated by the current checks
	 */
	public double getCoverage() {
		if (this.checked == 0) {
			return INITIAL_FITNESS;
		} else {
			return ((double) this.covered) / ((double) this.checked);
		}
	}
	
	public double getNs () {
		return updateStrategy.getNs(this);
	}

	
	public double getAccuracy () {
		return updateStrategy.getAccuracy(this);
	}
	
	
	/**
	 * Getter for the classifier's Serial Number.
	 * @return  the classifier's serial number
	 * @uml.property  name="serial"
	 */
	public int getSerial() {
		return this.serial;
	}

	/**
	 * Returns the data object saved at the classifier.
	 * 
	 * @return the update object
	 */
	public Serializable getUpdateDataObject() {
		return updateData;
	}

	
	public Serializable[] getUpdateDataArray() {
		return updateDataArray;
	}
	
	/**
	 * Get the string representation of the update-specific data.
	 * 
	 * @return a string with the representation
	 */
	public String getUpdateSpecificData() {
		return updateStrategy.getData(this);
	}

	
	/**
	 * Through the update strategy inherit the parameters
	 * 
	 * @param parentA
	 *            the first parent
	 * 
	 * @param parentB
	 *            the second parent
	 */
	public final void inheritParametersFromParents(Classifier parentA,
			Classifier parentB) {
		updateStrategy.inheritParentParameters(parentA, parentB, this);
	}

	
	
	/**
	 * Calls the bridge to detect if the classifier is matching the vision
	 * vector.
	 * 
	 * @param visionVector
	 *            the vision vector to match
	 * @return true if the classifier matches the visionVector
	 */
	public boolean isMatch(final double[] visionVector) {
		return transformBridge.isMatch(visionVector, this);
	}

	/**
	 * Checks if Classifier is matches an instance vector. Through caching for
	 * performance optimization.
	 * 
	 * @param instanceIndex
	 *            the instance index to check for a match
	 * @return true if the classifier matches the instance of the given index
	 */
	public boolean isMatch(final int instanceIndex) {
		if (this.matchInstances == null) {  // an einai kenos o pinakas matchInstances
			
			/*
			 * orizei ton pinaka matchInstances me mege9os {instances.length} 
			 * kai ton arxikopoiei me timi -1 se ka9e 9esi
			 * */
			buildMatches(); 
		}
		unmatched = 0;
		// if we haven't cached the answer, then answer...
		if (this.matchInstances[instanceIndex] == -1) {
			this.matchInstances[instanceIndex] = (byte) ((transformBridge.isMatch(myLcs.instances[instanceIndex], this)) ? 1 : 0);
			this.checked++; // ok o kanonas exei apofan9ei gia to instance (to exei dei) 
			this.covered += this.matchInstances[instanceIndex];
			
			/* would be kodikas
			 * 
			 * final boolean zeroCoverage = (this.getCheckedInstances() >= myLcs.instances.length) && (this.getCoverage() == 0);
			if(zeroCoverage) {
				//TODO
			}*/
			unmatched = 1;

		}

		return this.matchInstances[instanceIndex] == 1;
	}

	
	
	
	
	
	
	
	public boolean isMatchUnCached (final int instanceIndex) {
		
		this.matchInstances[instanceIndex] = (byte) ((transformBridge.isMatch(myLcs.instances[instanceIndex], this)) ? 1 : 0);
		this.checked++;
		this.covered += this.matchInstances[instanceIndex];
		
		if (this.checked == this.getLCS().instances.length) 
			this.objectiveCoverage = this.getCoverage();
		
		
		return this.matchInstances[instanceIndex] == 1;
	}
	
	
	
	public boolean isMatchCached(final int instanceIndex) {
		return this.matchInstances[instanceIndex] == 1;
	}
	
	
	
	public void buildMatchesForNewClassifier() {
		this.matchInstances = new byte[myLcs.instances.length];
	}
	
	
	
	
	
	
	
	
	
	
	/**
	 * Return if this classifier is more general than the testClassifier.
	 * 
	 * @param testClassifier
	 *            the test classifier
	 * @return true if the classifier is more general
	 */
	public boolean isMoreGeneral(final Classifier testClassifier) {
		return transformBridge.isMoreGeneral(this, testClassifier);
	}

	/**
	 * Setter for advocated action.
	 * 
	 * @param action
	 *            the action to set the classifier to advocate for
	 */
	public void setActionAdvocated(final int action) {
		transformBridge.setClassification(this, action);
		actionCache = null;
	}

	
	/**
	 * 
	 * Setter for the origin (cover or ga) of a classifier. 
	 * 
	 */
	public void setClassifierOrigin(int origin) {
		this.origin = origin;
	}
	
	
	/**
	 * Call the update strategy for setting value.
	 * 
	 * @param mode
	 *            the mode to set
	 * @param comparisonValue
	 *            the comparison value to set
	 */
	public void setComparisonValue(final int mode, final double comparisonValue) {
		updateStrategy.setComparisonValue(this, mode, comparisonValue);
	}

	/**
	 * Sets the update-specific and transform-specific data needed.
	 */
	private void setConstructionData() {
		
		/*
		 * stin ousia dimiourgo ena kainourio antikeimeno gia na krata ta dedomena enos classifier,
		 * ennoontas tis metablites fitness, ns, msa, tp, fp kai str kai mono. 
		 * arxikopoiei: fitness 0.5, ns = msa = tp = fp = str = 0
		 * */
		if (updateStrategy != null) {
			updateData = updateStrategy.createStateClassifierObject();
			updateDataArray = updateStrategy.createClassifierObjectArray();
		}

		if (transformBridge != null)
			transformBridge.setRepresentationSpecificClassifierData(this);

		this.serial = currentSerial; // ksekinaei me ton mikrotero int gia ton proto classifier kai auksanei
		currentSerial++;
	}

	/**
	 * Sets the classifier's LCS
	 * 
	 * @param lcs
	 *            the LCS
	 */
	public final void setLCS(AbstractLearningClassifierSystem lcs) {
		myLcs = lcs;
		updateStrategy = myLcs.getUpdateStrategy();
		transformBridge = myLcs.getClassifierTransformBridge();
	}

	/**
	 * Sets the classifier's subsumption ability.
	 * 
	 * @param canSubsumeAbility
	 *            true if the classifier is able to subsume
	 */
	public void setSubsumptionAbility(final boolean canSubsumeAbility) {
		subsumes = canSubsumeAbility;
	}

	/**
	 * Calls the bridge to divide bits into attributes.
	 * 
	 * @return the bitstring representation of the classifier
	 */
	public String toBitString() {
		return transformBridge.toBitSetString(this);
	}

	/**
	 * Calls the bridge to convert it self to natural language string.
	 * 
	 * @return the classifier described in a string
	 */
	@Override
	public String toString() {
		return transformBridge.toNaturalLanguageString(this);
	}

}