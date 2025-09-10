## **Unit 1: Basics of Probability, Random Processes & Pattern Recognition**

### **Q1. What is pattern recognition? List out various applications of pattern recognition. Discuss any one in detail.**

**Definition:**
Pattern Recognition (PR) is a scientific discipline that deals with the classification or description of observations (patterns) into categories or classes.

*   A **pattern** can be an image, signal, text, speech, or even a sequence of numbers.
*   The goal is to extract meaningful information and **automatically recognize similarities or differences** between inputs.
*   It forms the basis of **Artificial Intelligence (AI), Machine Learning (ML), and Data Science.**

**Tasks in Pattern Recognition:**

1.  **Data Acquisition:** Collect raw input (e.g., camera image).
2.  **Preprocessing:** Clean and normalize (e.g., remove noise, resize image).
3.  **Feature Extraction:** Select informative properties (e.g., edges in images, MFCC in speech).
4.  **Classification:** Apply a classifier (e.g., Bayes, SVM, Neural Networks).
5.  **Post Processing:** Verify, refine, and correct results.

**Applications of Pattern Recognition:**

1.  **Biometrics:** Fingerprint, iris, face recognition for identity verification.
2.  **Medical Diagnosis:** Detecting tumors or diseases from MRI/CT scans.
3.  **Speech Recognition:** Converting spoken words into text.
4.  **Optical Character Recognition (OCR):** Reading printed or handwritten documents.
5.  **Finance:** Stock market analysis, fraud detection in banking.
6.  **Autonomous Vehicles:** Object detection, pedestrian recognition.
7.  **Robotics & Automation:** Object manipulation, quality inspection.

**Detailed Example – Fingerprint Recognition (Biometrics):**

1.  **Input:** A fingerprint image is captured by a scanner.
2.  **Preprocessing:** The system performs noise removal, ridge enhancement, and thinning of lines to improve the image quality.
3.  **Feature Extraction:** Minutiae points, such as ridge endings and bifurcations, are extracted from the processed image.
4.  **Classification:** The extracted features are matched with a database of fingerprints using similarity measures.
5.  **Decision:** The identity is either verified or not verified based on the matching score.

### **Q2. Define supervised, unsupervised and reinforcement learning techniques.**

**1. Supervised Learning:**

*   **Definition:** Learning with the help of labeled training data (input-output pairs).
*   **Model:** Learns a mapping function: `f: X → Y`
*   **Examples:**
    *   Email Spam Detection (Input = email text, Output = spam/not spam).
    *   Handwritten digit recognition.
*   **Algorithms:** Linear Regression, Logistic Regression, Decision Trees, Neural Networks.

**2. Unsupervised Learning:**

*   **Definition:** Learning without labels. Only input data is given, and the system tries to find a hidden structure.
*   **Goal:** Clustering or grouping of data.
*   **Examples:**
    *   Market basket analysis in retail.
    *   Customer segmentation.
*   **Algorithms:** K-means, Hierarchical clustering, PCA.

**3. Reinforcement Learning (RL):**

*   **Definition:** An agent interacts with an environment, takes actions, receives rewards/penalties, and improves its strategy.
*   **Goal:** Maximize long-term cumulative reward.
*   **Examples:**
    *   Self-driving cars.
    *   Chess-playing AI.
    *   Robot navigation.
*   **Algorithms:** Q-Learning, Deep Q Networks (DQN).

**Comparison Table:**

| Aspect | Supervised | Unsupervised | Reinforcement |
| --- | --- | --- | --- |
| **Data** | Labeled | Unlabeled | No direct labels, only rewards |
| **Goal** | Predict output | Discover structure | Learn optimal policy |
| **Example** | Spam email detection | Customer clustering | Game-playing agent |

### **Q3. Explain important tasks/steps in pattern recognition with suitable diagram.**

**Steps in Pattern Recognition:**

1.  **Sensing (Data Acquisition):**
    *   Collects input data (image, sound, sensor signals).
    *   *Example:* A microphone records speech.

2.  **Preprocessing:**
    *   Involves data cleaning, scaling, and noise removal.
    *   *Example:* Image enhancement using filters.

3.  **Segmentation (if required):**
    *   Breaks data into meaningful parts.
    *   *Example:* Separating characters in handwritten text.

4.  **Feature Extraction:**
    *   Selects relevant properties (color, shape, edges).
    *   Reduces the dimensionality of the input.

5.  **Classification:**
    *   Applies algorithms like Bayes, k-NN, SVM, and Neural Networks.
    *   Assigns data to a class.

6.  **Post-Processing & Decision:**
    *   Improves output and handles ambiguities.
    *   *Example:* Spell-checker in OCR.

**Diagram (Block Representation):**

`Raw Data → Preprocessing → Segmentation → Feature Extraction → Classification → Output`


### **Q4. Explain joint and conditional probability with example.**

**Joint Probability:**
*   It is the probability of two events happening together.
*   **Formula:** P(A ∩ B) = Probability(A and B)
*   **Example:** Tossing two dice.
    *   Event A: Die 1 shows 2.
    *   Event B: Die 2 shows 5.
    *   The joint probability P(A ∩ B) is 1/36, as there is only one outcome (2, 5) out of 36 possible outcomes.

**Conditional Probability:**
*   It is the probability of event A occurring given that event B has already occurred.
*   **Formula:** P(A|B) = P(A ∩ B) / P(B)
*   **Example:** Drawing a card from a deck of 52 cards.
    *   What is the probability that the card is a King, given that it is a face card?
    *   Face cards = 12 (J, Q, K of 4 suits)
    *   Kings = 4
    *   P(King | Face Card) = (Number of Kings that are Face Cards) / (Total Number of Face Cards) = 4/12 = 1/3.

**Importance in Pattern Recognition:**
*   Bayes' theorem, a fundamental concept for classification, is built on conditional probability.

### **Q5. Explain the terms: Autocorrelation, Cross-correlation, Autocovariance.**

1.  **Autocorrelation:**
    *   **Definition:** Measures the similarity of a signal with a delayed copy of itself (at different time lags).
    *   **Formula:** Rₓₓ(τ) = E[x(t) * x(t + τ)]
    *   **Example:** Detecting repeating patterns in ECG signals to identify heart rate.

2.  **Cross-correlation:**
    *   **Definition:** Measures the similarity between two different signals as a function of the time lag applied to one of them.
    *   **Formula:** Rₓᵧ(τ) = E[x(t) * y(t + τ)]
    *   **Example:** Comparing a transmitted signal versus a received signal in communication systems to determine the time delay.

3.  **Autocovariance:**
    *   **Definition:** Measures how a signal varies with itself over time, taking the mean into account.
    *   **Formula:** Cₓₓ(τ) = E[(x(t) - μ)(x(t + τ) - μ)]
    *   **Example:** Used in time series forecasting to understand the persistence of shocks or trends.

### **Q6. Probability Problem (Machine Operators A, B, C)**

**Given:**
*   Operator A: Works 50%, defect rate = 1%
*   Operator B: Works 30%, defect rate = 5%
*   Operator C: Works 20%, defect rate = 7%
*   An item is chosen randomly and is defective. Find the probability it was made by A.

**Solution using Bayes' Theorem:**
*   Let D be the event that the item is defective. We want to find P(A|D).
*   **Formula:** P(A|D) = [P(D|A) * P(A)] / P(D)
    *   Where P(D) = P(D|A)P(A) + P(D|B)P(B) + P(D|C)P(C)

**Substitute Values:**
*   P(A) = 0.50, P(D|A) = 0.01 → P(D|A)P(A) = 0.005
*   P(B) = 0.30, P(D|B) = 0.05 → P(D|B)P(B) = 0.015
*   P(C) = 0.20, P(D|C) = 0.07 → P(D|C)P(C) = 0.014
*   Denominator (P(D)) = 0.005 + 0.015 + 0.014 = 0.034

**Calculation:**
*   P(A|D) = 0.005 / 0.034 ≈ 0.147

**Result:** The probability that the defective item was made by Operator A is **14.7%**.



---

## **Unit 2: Eigenvalues and Eigenvectors**

### **Q-1. Eigenvalues and Eigenvectors**

**(a) Definition**

Let *A* be a square matrix of size n x n. A non-zero vector *v* ∈ Rⁿ is called an **eigenvector** of *A* if there exists a scalar λ ∈ R (or C) such that:

*Av = λv*

Here,
*   *v* = eigenvector
*   λ = eigenvalue

This means that when a matrix acts on its eigenvector, the direction does not change — only the length (scaled by λ) changes.

**(b) Characteristic Equation**

To find eigenvalues, solve:

det(*A* – λ*I*) = 0

This gives the characteristic polynomial, whose roots are the eigenvalues.

**(c) Geometric Meaning**

*   Matrices usually rotate, stretch, or flip vectors.
*   Eigenvectors are "special directions" where the transformation only stretches/compresses but doesn't change the direction.
*   Eigenvalues are the scale factors.
    *   **Example:** Stretching an ellipse—eigenvectors point to the ellipse's principal axes, and eigenvalues represent how much stretching occurs.

**(d) Properties of Eigenvalues & Eigenvectors**

1.  **Sum of eigenvalues** = Trace(A).
2.  **Product of eigenvalues** = Determinant(A).
3.  If *A* is symmetric, eigenvalues are real, and eigenvectors of different eigenvalues are orthogonal.
4.  Eigenvectors are not unique—any non-zero scalar multiple is valid.
5.  If *A* is invertible, none of the eigenvalues is zero.
6.  A matrix is diagonalizable if it has *n* linearly independent eigenvectors.

### **Q-2. Singular Value Decomposition (SVD)**

**(a) Definition**

Every real matrix *A* ∈ R<sup>mxn</sup> can be written as:

A = UΣV<sup>T</sup>

where:
*   **U** = orthogonal *m x m* matrix (left singular vectors)
*   **Σ** = diagonal *m x n* matrix (singular values σ₁ ≥ σ₂ ≥ … ≥ 0)
*   **V** = orthogonal *n x n* matrix (right singular vectors)

**(b) Steps to Compute**

1.  Compute A<sup>T</sup>A.
2.  Find the eigenvalues of A<sup>T</sup>A.
    *   Singular values: σᵢ = √λᵢ
    *   Eigenvectors: right singular vectors (V).
3.  Compute U using uᵢ = (1/σᵢ)Avᵢ.
4.  Form Σ = diag(σ₁, σ₂, …).

**(c) Applications of SVD**

1.  **Principal Component Analysis (PCA):** Dimension reduction.
2.  **Image Compression:** Keep only the largest singular values.
3.  **Noise Reduction:** Discard small singular values.
4.  **Recommender Systems:** Netflix and Amazon use SVD.
5.  **Solving ill-conditioned systems:** For numerical stability.


### **Q3. Eigenvalues vs Singular Values (Important Difference Question)**

| Feature | Eigenvalues/Eigenvectors | Singular Values (SVD) |
| :--- | :--- | :--- |
| **Defined for** | Square matrices only | Any rectangular/square matrix |
| **Can be negative?** | Yes, even complex | Always non-negative real |
| **Formula** | Av = λv | σ = √eigenvalue of AᵀA |
| **Decomposition**| A = PDP⁻¹ if diagonalizable | A = UΣVᵀ always exists |
| **Vectors** | Eigenvectors may not be orthogonal | U and V are always orthogonal |
| **Applications** | Stability analysis, differential equations | PCA, compression, pseudoinverse |
| **Special Case** | If A is a symmetric positive semidefinite matrix, its eigenvalues are equal to its singular values. | |

---

## **Unit 3: Bayesian Learning**

### **Q-1. Describe Bayesian learning in pattern recognition.**

Bayesian learning is a statistical approach to pattern recognition where probability theory is used to make decisions in the presence of uncertainty.

*   In pattern recognition, we have different possible classes (ω₁, ω₂, ..., ωₙ).
*   Each class represents a category into which an observed sample (feature vector *x*) may fall.
*   Bayesian learning applies Bayes' theorem to compute the posterior probability of each class given the observed data.

**Bayes' Theorem:**

P(ωᵢ|x) = [P(x|ωᵢ) * P(ωᵢ)] / P(x)

Where:

*   **P(ωᵢ)** = **Prior probability** of class *i*.
*   **P(x|ωᵢ)** = **Likelihood** of observing *x* given class *i*.
*   **P(x)** = **Evidence** or total probability of observing *x*.
*   **P(ωᵢ|x)** = **Posterior probability** of class *i* after observing data.

**Decision Rule:** Assign *x* to the class with the maximum posterior probability (MAP rule).

### **Q-2. Write a short note on Minimum error rate classification.**

The goal of pattern recognition is to minimize classification errors. Minimum-error-rate classification is a Bayesian decision strategy where we assign a pattern to the class with the highest posterior probability.

**Rule:**
For a feature vector *x*, decide class ωᵢ if:
P(ωᵢ|x) > P(ωⱼ|x), for all j ≠ i

This rule minimizes the probability of making a wrong decision.

**Error Probability:**
The probability of error is:
P(error) = 1 - maxᵢ[P(ωᵢ|x)]

**Intuitive Example:**
Suppose we want to classify whether a coin toss outcome belongs to a "fair coin" or a "biased coin".
*   Posterior probabilities are calculated for both cases.
*   If the posterior for "biased coin" is higher, we classify it as biased.
*   This ensures a minimum chance of being wrong.

Hence, this rule is also known as the **Bayes Optimal Classifier.**


### **Q3. Give the expression and explain all parameters of univariate and multivariate normal density in context of Bayesian decision theory.**

**(a) Univariate Normal Density**
*   **Expression:** p(x | μ, σ²) = (1 / √(2πσ²)) * exp(- (x - μ)² / (2σ²))
*   **Parameters:**
    *   **x:** Observed scalar feature.
    *   **μ:** Mean of the distribution.
    *   **σ²:** Variance (spread) of the distribution.
*   **Use in Bayesian Theory:** Models the likelihood of observing a feature given a class. For example, the height of people can be modeled by a univariate Gaussian.

**(b) Multivariate Normal Density**
*   **Expression:** p(x | μ, Σ) = (1 / ((2π)^(d/2) |Σ|^(1/2))) * exp(-½ (x - μ)ᵀ Σ⁻¹ (x - μ))
*   **Parameters:**
    *   **x:** Feature vector of dimension *d*.
    *   **μ:** Mean vector (*d* x 1).
    *   **Σ:** Covariance matrix (*d* x *d*).
    *   **|Σ|:** Determinant of the covariance matrix.
    *   **Σ⁻¹:** Inverse of the covariance matrix.
*   **Geometric Interpretation:** The density defines an elliptical distribution. Eigenvectors of Σ define the orientation, while eigenvalues define the spread. Used in classifiers like QDA and LDA.

### **Q4. Apply Bayes' Minimum Risk Classifier.**

**(Based on the data in the PDF)**
**Given:** P(ω1)=0.3, P(ω2)=0.7, P(x|ω1)=0.65, P(x|ω2)=0.5. Assume a 0-1 loss matrix (equal losses).

**Step 1: Calculate Likelihood Ratio**
*   Λ(x) = P(x|ω1) / P(x|ω2) = 0.65 / 0.5 = 1.3

**Step 2: Calculate Decision Threshold**
*   Assuming 0-1 loss, the threshold is P(ω2) / P(ω1).
*   Threshold = 0.7 / 0.3 ≈ 2.33

**Step 3: Apply Decision Rule**
*   The rule is: Classify as ω1 if Λ(x) > Threshold, otherwise classify as ω2.
*   Since Λ(x) = 1.3 < 2.33, the decision is to **classify as ω2**.

**Step 4: Compute Bayes Risk**
*   The risk for choosing ω2 is R(α₂|x) = λ₂₁P(ω₁|x), where λ₂₁ is the loss for deciding ω2 when the true class is ω1 (assumed to be 1).
*   First, find the posterior P(ω₁|x) = [P(x|ω₁)P(ω₁)] / P(x).
    *   Given P(x) = 0.545 in the PDF.
    *   P(ω₁|x) = (0.65 * 0.3) / 0.545 ≈ 0.3587
*   The Bayes risk is **R(α₂|x) ≈ 0.3587**.

**Final Decision:** Choose ω2, with a Bayes risk of approximately 0.36.

### **Q5. Explain "Maximum a Posteriori” (MAP) estimation.**

MAP estimation is a Bayesian approach to classification that assigns a data sample *x* to the class ωᵢ for which the posterior probability P(ωᵢ|x) is maximum.

*   **Formula:** Class = arg maxᵢ P(ωᵢ|x)
*   From Bayes' theorem, P(ωᵢ|x) ∝ P(x|ωᵢ)P(ωᵢ). Since P(x) is constant across all classes, it can be ignored for maximization.
*   **MAP combines:**
    1.  **Prior probability P(ωᵢ):** Our belief about the class before seeing the data.
    2.  **Likelihood P(x|ωᵢ):** How well the data fits the class.
*   **Conclusion:** MAP gives a more reliable classification than using only the likelihood (Maximum Likelihood method) because it incorporates prior knowledge.

### **Q6. Describe linearly separable and linearly inseparable classification problems.**

*   **Linearly Separable:**
    *   A dataset is linearly separable if a single straight line (2D), plane (3D), or hyperplane (nD) can separate samples of different classes without any errors.
    *   **Example:** Classifying students based on height and weight might be linearly separable if one class is consistently taller and heavier.
    *   Many classifiers like the Perceptron and linear SVM work well on this type of data.

*   **Linearly Inseparable:**
    *   If no such linear boundary exists, the classes overlap.
    *   To classify this data, non-linear classifiers are needed, such as SVM with a kernel, neural networks, or decision trees.
    *   **Example:** The XOR problem, where points (0,1) and (1,0) belong to one class and (0,0) and (1,1) to another, cannot be separated by a single straight line.

### **Q7. Discuss different cases of Bayesian parameter estimation.**

In Bayesian parameter estimation, parameters are treated as random variables with probability distributions.

1.  **Case 1: Known Prior, Known Likelihood:** The posterior distribution of the parameter θ is calculated using Bayes' theorem: P(θ|x) ∝ P(x|θ)P(θ).
2.  **Case 2: Conjugate Priors:** Special priors are chosen so that the posterior distribution belongs to the same family as the prior. This simplifies computation.
    *   *Example:* A Gaussian likelihood combined with a Gaussian prior results in a Gaussian posterior.
3.  **Case 3: Non-informative Priors:** A prior that conveys no information (e.g., a uniform prior) is used. In this case, the estimation often reduces to Maximum Likelihood Estimation (MLE).
4.  **Case 4: Hierarchical Bayes:** Priors themselves have hyperparameters, which are also given distributions. This is useful for complex models.

### **Q8. Prove that a Bayes classifier is equivalent to a minimum distance classifier, assuming the feature vector is Gaussian.**

**Step 1: Bayes Classifier Rule**
*   Choose class ωᵢ if: P(ωᵢ|x) > P(ωⱼ|x) for all j ≠ i.
*   This is equivalent to: p(x|ωᵢ)P(ωᵢ) > p(x|ωⱼ)P(ωⱼ).

**Step 2: Log-likelihood Comparison with Gaussian Assumption**
*   Assuming Gaussian class-conditional densities and taking the logarithm, the decision rule depends on a discriminant function gᵢ(x).
*   gᵢ(x) = log(p(x|ωᵢ)) + log(P(ωᵢ))
*   Substituting the Gaussian density, and removing terms common to all classes, we get:
    *   gᵢ(x) = -½ (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ) - ½ log|Σᵢ| + log P(ωᵢ)

**Step 3: Simplify with Further Assumptions**
*   If we assume that **priors are equal** (P(ωᵢ) = P(ωⱼ)) and the **covariance matrices are equal and spherical** (Σᵢ = σ²I), the terms involving log P(ωᵢ) and the covariance matrix become constant across all classes.
*   The discriminant function simplifies to:
    *   gᵢ(x) ∝ -||x - μᵢ||²
*   Maximizing gᵢ(x) is equivalent to minimizing ||x - μᵢ||², which is the squared Euclidean distance.

**Conclusion:** Under the assumptions of Gaussian distributions, equal priors, and equal spherical covariance matrices, the Bayes classifier becomes a **Minimum Euclidean Distance Classifier**.

### **Q9. What is Bayes theorem? Describe its use in classification.**

**Bayes' Theorem:**
A fundamental rule in probability that describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
*   **Formula:** P(ωᵢ|x) = [P(x|ωᵢ) * P(ωᵢ)] / P(x)

**Use in Classification:**
1.  **Compute Posterior:** For a given data point *x*, the theorem is used to compute the posterior probability P(ωᵢ|x) for each possible class ωᵢ.
2.  **Assign Class:** The data point *x* is assigned to the class with the maximum posterior probability.
3.  **Minimize Misclassification:** This approach is Bayes optimal, meaning it minimizes the probability of misclassification.
*   **Example:** In handwriting recognition, Bayes' theorem helps decide if an image is a '3' or an '8' based on the prior frequency of those digits and the likelihood of the pixel patterns for each digit.


---

## **Unit 4: Expectation-Maximization and Clustering**

### **Q-1. Discuss the working of the Expectation-Maximization algorithm.**

The Expectation-Maximization (EM) algorithm is an iterative method for finding maximum likelihood estimates of parameters when the data is incomplete, hidden, or has missing values.

**Steps of the EM Algorithm:**

1.  **Initialization:** Assume initial parameters (mean, variance, mixture weights).
2.  **E-step (Expectation):** Estimate membership probabilities of each data point for hidden variables based on the current parameters.
3.  **M-step (Maximization):** Update parameters (means, variances, weights) by maximizing the expected likelihood.
4.  **Repeat:** Continue the E-step and M-step until convergence.

**Applications of EM:**

1.  **Clustering:** GMM, K-means extension.
2.  **Image Processing:** Denoising, segmentation.
3.  **Speech Recognition:** Training HMMs.
4.  **Medical Imaging:** MRI reconstruction.

### **Q-2. Discuss Hierarchical clustering in detail.**

Hierarchical clustering is a clustering method that builds a hierarchy of clusters. Unlike K-means, it does not require specifying the number of clusters in advance. It produces a **dendrogram**, which is a tree-like structure of clusters.

**Types of Hierarchical Clustering:**

1.  **Agglomerative (Bottom-Up):**
    *   Starts with each object as a separate cluster.
    *   At each step, merges the two closest clusters until all points belong to a single cluster.

2.  **Divisive (Top-Down):**
    *   Starts with one cluster containing all objects.
    *   Splits clusters recursively until each point is separate.

**Linkage Criteria (Distance Between Clusters):**

*   **Single Linkage:** Minimum distance between any two points of different clusters.
*   **Complete Linkage:** Maximum distance between any two points.
*   **Average Linkage:** Average distance between all pairs of points.
*   **Centroid Linkage:** Distance between cluster centroids.

**Advantages:**

*   Does not need the number of clusters beforehand.
*   Produces a full cluster hierarchy.
*   Works with different distance measures.

**Disadvantages:**

*   Computationally expensive O(n³).
*   Sensitive to noise and outliers.
*   Once merged/split, cannot be undone.


### **Q3. What is the significance of Hidden Markov Models in classifier design?**

A Hidden Markov Model (HMM) is a probabilistic model for sequential data where the system is modeled as hidden states that emit observable outputs.

**Significance in Classifier Design:**
1.  **Sequential Data Modeling:** HMMs are excellent at capturing temporal dependencies in data, such as speech signals, handwriting, or DNA sequences.
2.  **Probabilistic Framework:** They compute the likelihood of an observed sequence for a given model, which allows for robust classification. For example, P(observations | word model).
3.  **Versatility:** They can model various time series, gestures, and biological sequences.
*   **Example (Speech Recognition):** Each word ("yes", "no") is modeled by a separate HMM. When a user speaks, the system computes the likelihood of the speech signal for each word's HMM and chooses the word with the maximum likelihood.

### **Q4. What is k-NN classification algorithm?**

The k-Nearest Neighbor (k-NN) algorithm is a non-parametric, instance-based learning method for classification and regression. It classifies a new data point based on the majority class among its *k* nearest neighbors in the training dataset.

**Steps of k-NN Algorithm:**
1.  **Choose k:** Select the number of neighbors (e.g., k=3 or k=5).
2.  **Compute Distance:** For a new test sample, calculate the distance (e.g., Euclidean) to every training sample.
    *   Euclidean distance d(x, zᵢ) = √Σ(xⱼ - zᵢⱼ)²
3.  **Find k Nearest Neighbors:** Sort all training samples by their distance to the test sample and select the closest *k*.
4.  **Majority Voting Rule:** Count the class labels of these *k* neighbors. The test point is assigned to the class that occurs most frequently.

### **Q5. Discuss various criterion functions for clustering.**

Criterion functions are mathematical measures used to evaluate the quality of a clustering solution.

1.  **Minimum Squared Error Criterion (MSE):**
    *   Used in K-means clustering.
    *   **Goal:** Minimize the sum of squared distances between data points and their cluster centroids. A lower MSE indicates more compact clusters.
    *   **Formula:** J = Σᵢ Σ_{x ∈ Cᵢ} ||x - μᵢ||²

2.  **Maximum Separation Criterion:**
    *   **Goal:** Maximize the distance between cluster centroids, ensuring clusters are well-separated.
    *   **Formula:** D = min_{i≠j} ||μᵢ - μⱼ||

3.  **Entropy-based Criterion:**
    *   Measures the impurity (mixture of classes) inside clusters.
    *   A lower entropy indicates purer clusters.

4.  **Silhouette Coefficient:**
    *   Combines cohesion (within-cluster similarity) and separation (between-cluster dissimilarity).
    *   A value close to 1 indicates good clustering.

5.  **Dunn Index:**
    *   Ratio of the minimum inter-cluster distance to the maximum intra-cluster diameter.
    *   A high Dunn Index indicates compact and well-separated clusters.

### **Q6. Discuss Non-negative Matrix Factorization (NMF).**

NMF is a dimensionality reduction and feature extraction technique that factors a non-negative data matrix *X* into two smaller non-negative matrices, *W* and *H*.

*   **Mathematical Formulation:** X ≈ WH
    *   **X (m x n):** Original data matrix.
    *   **W (m x r):** Basis matrix (dictionary).
    *   **H (r x n):** Coefficients (encoding data in terms of basis).
    *   **Constraint:** W ≥ 0, H ≥ 0.
*   **Interpretation:** NMF provides interpretable features because all values are non-negative. For example, in topic modeling, *W* contains topics (basis words) and *H* contains document-topic weights.
*   **Applications:**
    *   Topic modeling in text mining.
    *   Image decomposition (parts-based learning).
    *   Recommender systems.

### **Q7. Explain the nonparametric methods for density estimation.**

Nonparametric methods estimate the probability distribution of data without making any assumptions about the underlying distribution (e.g., assuming it's Gaussian).

1.  **Histogram Method:**
    *   Divides the data range into bins.
    *   Counts the number of samples in each bin.
    *   Probability density = (frequency) / (total samples * bin width).

2.  **Kernel Density Estimation (KDE):**
    *   Uses a smooth kernel function (like a Gaussian) instead of hard bins.
    *   A kernel is placed over each data point, and the sum of all kernels forms the density estimate.
    *   The bandwidth (*h*) parameter controls the smoothness of the curve.

3.  **k-Nearest Neighbor (k-NN) Density Estimation:**
    *   The volume around a point is adjusted until it includes *k* samples.
    *   Density = k / (n * V), where *n* is the total number of samples and *V* is the volume.

### **Q7 (Duplicate). Discuss different cluster validation methods.**

Cluster validation evaluates the quality of clustering results.

1.  **Internal Validation (uses data only):**
    *   Measures the quality based on the inherent structure of the data.
    *   **Methods:** Silhouette Coefficient (balances cohesion and separation), Dunn Index (ratio of inter-cluster to intra-cluster distances).
    *   **Goal:** Compact and separated clusters.

2.  **External Validation (uses ground truth labels):**
    *   Compares the clustering results to pre-existing labels.
    *   **Methods:** Rand Index (compares similarity between predicted and true clusters), Purity (fraction of correctly assigned samples).
    *   **Goal:** Agreement with ground truth.

3.  **Relative Validation:**
    *   Compares different clustering algorithms or different settings of the same algorithm (e.g., varying *k* in K-means).
    *   Chooses the one with the best internal/external scores.

### **Q8. Differentiate between Discrete HMM and Continuous HMM.**

| Feature | Discrete HMM | Continuous HMM |
| :--- | :--- | :--- |
| **Observation Symbols** | Finite, categorical symbols (e.g., words, phonemes). | Continuous values (e.g., speech signals, sensor readings). |
| **Emission Probability** | A probability mass function, bⱼ(oₖ). | A probability density function, bⱼ(o), often a Gaussian mixture. |
| **Modeling** | Simple, used for categorical data. | Complex, models real-valued data. |
| **Applications** | Text classification, DNA sequencing. | Speech recognition, handwriting recognition. |


---

## **Unit 8: Decision Tree Learning**

### **Q.1 Illustrate the working of a decision tree learning algorithm.**

A decision tree is a supervised learning algorithm used for classification and prediction. It represents decisions in the form of a tree structure, where:

*   **Internal nodes** → represent tests on attributes.
*   **Branches** → represent outcomes of the test.
*   **Leaf nodes** → represent final class labels or decisions.

**Working Steps:**

1.  Start with the training dataset containing features and a target class.
2.  Select the **best attribute** for splitting the data using measures such as:
    *   Information Gain (Entropy-based)
    *   Gini Index
3.  Create a **decision node** based on this attribute.
4.  **Split the dataset** into subsets according to attribute values.
5.  **Repeat recursively** for each subset until:
    *   All records in a subset belong to the same class, or
    *   No attributes remain.
6.  The final **decision tree** is then used to classify new instances by tracing from the root to a leaf.

### **Q.2 What are the Advantages and Disadvantages of Decision Tree learning?**

**Advantages:**

1.  Rules are simple and easy to understand.
2.  Can handle both nominal and numerical attributes.
3.  Capable of handling datasets that may have errors or missing values.
4.  Considered to be a non-parametric method.
5.  Decision trees are self-explanatory.

**Disadvantages:**

1.  Most algorithms require that the target attribute has only discrete values.
2.  Some problems are difficult to solve, like XOR.
3.  Less appropriate for estimation tasks where the goal is to predict the value of a continuous variable.
4.  Prone to errors in classification problems with many classes and a relatively small number of training examples.

### **Q.3 What is Pruning? What is its significance?**

**Definition:**
Pruning is the process of removing unnecessary branches or nodes from a decision tree to reduce its size and complexity.

**Types of Pruning:**

1.  **Pre-pruning (Early Stopping):** Stop tree growth early if further splitting does not improve accuracy.
2.  **Post-pruning:** Grow the full tree first, then remove branches that do not improve performance.

**Significance of Pruning:**

*   **Reduces Overfitting:** Makes the model more general and accurate for unseen data.
*   **Simplifies the Tree:** Easier to understand and interpret.
*   **Improves Accuracy:** Removes noisy or irrelevant splits.
*   **Faster Prediction:** A smaller tree leads to quicker classification.


### **Q4. When does a Decision Tree require pruning? How can pruning be done?**

**A decision tree requires pruning when:**
1.  The tree becomes very large and complex.
2.  It starts to **overfit** the training data.
3.  Some branches are based on noise or irrelevant attributes.
4.  The accuracy on unseen/test data decreases.

**How pruning can be done:**
1.  **Pre-pruning (Early Stopping):**
    *   Stops tree growth before it becomes too complex.
    *   **Conditions Used:** Limit maximum depth, set a minimum number of samples required at a node, or require a minimum information gain threshold to split.

2.  **Post-pruning (Simplifying After Full Growth):**
    *   Grows the complete tree first, then removes weak/unnecessary branches.
    *   **Methods:**
        *   **Reduced Error Pruning:** Prune a branch if it does not reduce accuracy on a validation set.
        *   **Cost Complexity Pruning:** Prune branches by considering a trade-off between accuracy and tree size.

### **Q5. Explain the problem of “overfitting” in decision tree. How can this problem be overcome?**

**Problem of Overfitting:**
*   Overfitting occurs when a decision tree becomes too large and complex, trying to fit all details, including noise and outliers, of the training data.
*   Such a tree performs very well on training data but poorly on test/unseen data (poor generalization).

**How to Overcome Overfitting:**
1.  **Pruning the Tree:**
    *   **Pre-pruning:** Stop tree growth early.
    *   **Post-pruning:** Grow the full tree, then remove weak branches.
2.  **Use of Validation Set:**
    *   Evaluate performance on unseen validation data while building the tree.
    *   Stop splitting if accuracy on the validation set does not improve.
3.  **Ensemble Methods:**
    *   Use techniques like **Random Forests** or **Boosting**, which combine multiple trees to reduce overfitting and improve robustness.

### **Q6. Discuss Decision Tree learning based on the CART approach.**

**CART (Classification and Regression Trees):**
*   CART is a decision tree algorithm that can handle both classification (categorical output) and regression (continuous output) problems.

**Working of CART:**
1.  **Tree Construction:**
    *   Starts with the training dataset.
    *   At each node, CART selects the best attribute and split point.
    *   The dataset is split into two child nodes (CART always produces **binary trees**).
2.  **Splitting Criterion:**
    *   **For Classification:** The **Gini Index** is used to measure impurity.
        *   Gini(D) = 1 - Σ(pᵢ)²
    *   **For Regression:** **Variance Reduction** or Mean Squared Error (MSE) is used.
3.  **Pruning:**
    *   CART uses **Cost Complexity Pruning** to avoid overfitting by balancing tree size versus accuracy.

**Features of CART:**
*   Always produces binary splits.
*   Uses Gini Index (classification) and MSE/variance (regression).
*   Handles both numeric and categorical attributes.
