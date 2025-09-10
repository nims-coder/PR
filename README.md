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

---
---

## MID IMP 
---

### **Q. 1**

**(a) What is “curse of dimensionality”? How to overcome this? (3 Marks)**

The **"curse of dimensionality"** refers to a set of problems that arise when analyzing data in high-dimensional spaces. As the number of features (dimensions) increases, the data becomes extremely sparse, and the volume of the space grows so fast that the available data becomes insufficient to fill it.

This leads to several issues:
*   The distance between any two points in a high-dimensional space becomes less meaningful.
*   Much more data is required to support a statistically sound conclusion.
*   The risk of model overfitting increases significantly.

**How to Overcome It:**
1.  **Feature Selection:** Select a subset of the most relevant original features and discard the rest.
2.  **Feature Extraction / Dimensionality Reduction:** Create new, lower-dimensional features by combining the original ones. Common techniques include:
    *   Principal Component Analysis (PCA)
    *   Linear Discriminant Analysis (LDA)

**(b) Find the eigenvalues and corresponding eigenvectors for the matrix A = [,]. (4 Marks)**

**Step 1: Find the Eigenvalues (λ)**
We solve the characteristic equation det(A - λI) = 0.
*   A - λI = `[[1-λ, 2], [4, 3-λ]]`
*   det(A - λI) = (1-λ)(3-λ) - (2)(4)
    = 3 - 4λ + λ² - 8
    = λ² - 4λ - 5
*   Set the equation to zero: λ² - 4λ - 5 = 0
*   Factor the quadratic equation: (λ - 5)(λ + 1) = 0
*   The eigenvalues are **λ₁ = 5** and **λ₂ = -1**.

**Step 2: Find the Eigenvectors (v)**
For each eigenvalue, we solve (A - λI)v = 0.

*   **For λ₁ = 5:**
    *   (A - 5I)v = `[[-4, 2], [4, -2]]` `[[x], [y]]` = `[[0], [0]]`
    *   This gives the equation -4x + 2y = 0, which simplifies to y = 2x.
    *   If we let x = 1, then y = 2.
    *   The corresponding eigenvector is **v₁ =ᵀ**.

*   **For λ₂ = -1:**
    *   (A - (-1)I)v = `[[2, 2], [4, 4]]` `[[x], [y]]` = `[[0], [0]]`
    *   This gives the equation 2x + 2y = 0, which simplifies to y = -x.
    *   If we let x = 1, then y = -1.
    *   The corresponding eigenvector is **v₂ = [1, -1]ᵀ**.

**(c) Write a short note on Minimum error rate classification. (7 Marks)**

Minimum error rate classification is a fundamental decision-making strategy in pattern recognition based on **Bayesian Decision Theory**. Its primary goal is to design a classifier that minimizes the total probability of making an incorrect classification (misclassification error).

**The Decision Rule:**
The core principle is to assign a feature vector **x** to the class **ωᵢ** that has the highest **posterior probability**, P(ωᵢ|x).
*   **Rule:** Decide ωᵢ if **P(ωᵢ|x) > P(ωⱼ|x)** for all j ≠ i.

**Bayes' Theorem:**
The required posterior probability is calculated using Bayes' theorem, which connects it to more easily estimated quantities: the prior and the likelihood.
*   **P(ωᵢ|x) = [ p(x|ωᵢ) * P(ωᵢ) ] / p(x)**
    *   **P(ωᵢ|x) (Posterior):** The probability of the class being ωᵢ *after* observing the data x.
    *   **p(x|ωᵢ) (Likelihood):** The probability of observing data x if it belonged to class ωᵢ. This is the class-conditional density.
    *   **P(ωᵢ) (Prior):** The probability of class ωᵢ occurring in general, *before* seeing any data.
    *   **p(x) (Evidence):** The overall probability of observing the data x. It acts as a normalizing constant.

**Minimizing Error:**
By always choosing the class with the highest posterior probability, we are inherently minimizing the error for each decision. The probability of error for any given decision is P(error|x) = 1 - maxᵢ P(ωᵢ|x). By maximizing the posterior, we minimize this error. A classifier that implements this rule is called a **Bayes Optimal Classifier**, as no other classifier can achieve a lower average error rate given the same prior and likelihood information.

---

### **Q. 2**

**(a) What is unsupervised learning? What is soft clustering and hard clustering? (3 Marks)**

*   **Unsupervised Learning:** A type of machine learning where algorithms learn patterns from unlabeled data. The system tries to find hidden structures or relationships within the data without any predefined target outputs. Common tasks are clustering and dimensionality reduction.
*   **Hard Clustering:** An approach where each data point is assigned exclusively to one cluster. The membership is absolute and non-overlapping. *Example: K-means clustering.*
*   **Soft Clustering (Fuzzy Clustering):** An approach where each data point is assigned a probability or a degree of membership for every cluster. A data point can belong to multiple clusters simultaneously, but with different levels of certainty. *Example: Gaussian Mixture Models (GMM).*

**(b) What is pattern recognition? List out various applications of pattern recognition. (4 Marks)**

**Pattern Recognition** is the scientific discipline focused on the automatic discovery of patterns, regularities, and structures in data. It is a core part of machine learning that enables systems to classify data into different categories. The process typically involves acquiring raw data, preprocessing it, extracting meaningful features, and then using a classification algorithm to assign a label or category to the input.

**Various Applications:**
*   **Computer Vision:** Face detection, object recognition, optical character recognition (OCR).
*   **Medical Diagnosis:** Analyzing medical images (like X-rays or MRIs) to detect tumors or diseases.
*   **Biometrics:** Fingerprint matching, iris scanning, and voice recognition for security and authentication.
*   **Speech Recognition:** Converting spoken words into text (e.g., Siri, Google Assistant).
*   **Finance:** Credit card fraud detection and stock market analysis.
*   **Natural Language Processing:** Spam filtering in emails and sentiment analysis on social media.

**(c) Explain the steps in "k-means clustering" using a suitable illustration. What are the limitations of this method? (7 Marks)**

K-means is a popular unsupervised learning algorithm used for partitioning a dataset into *k* distinct, non-overlapping clusters.

**Steps in k-means clustering:**
1.  **Initialization:** Choose the number of clusters, *k*. Then, randomly initialize *k* points as the initial cluster centroids.
2.  **Assignment Step:** For each data point in the dataset, calculate its distance (typically Euclidean distance) to all *k* centroids. Assign the data point to the cluster of the nearest centroid.
3.  **Update Step:** After all points have been assigned to clusters, recalculate the position of the *k* centroids. The new centroid of each cluster is the mean of all data points assigned to it.
4.  **Convergence:** Repeat the Assignment and Update steps iteratively until the cluster assignments no longer change or the centroids stop moving significantly.

**Illustration:**

| Step 1: Initialize | Step 2: Assign | Step 3: Update | Final Clusters |
| :---: | :---: | :---: | :---: |
|  |  |  |  |
| Three random centroids are placed. | Points are colored based on the closest centroid. | Centroids move to the mean of their new points. | The process repeats until centroids stabilize. |

**Limitations of k-means:**
1.  **Pre-defined *k***: The number of clusters, *k*, must be specified in advance, which is often difficult to determine.
2.  **Sensitivity to Initialization:** The final result can vary depending on the initial random placement of centroids.
3.  **Cluster Shape:** It assumes that clusters are spherical and evenly sized, performing poorly on clusters with complex, non-convex shapes or varying densities.
4.  **Sensitivity to Outliers:** Outliers can significantly pull a centroid away from its true center, affecting the final clustering.

**(c) OR Explain working of Hidden Markov Model. (7 Marks)**

A Hidden Markov Model (HMM) is a statistical model for analyzing sequential data where the underlying system is assumed to be a Markov process with unobserved (hidden) states.

**Core Components:**
1.  **Hidden States (S):** A finite set of states the system can be in, which are not directly observable (e.g., 'Hot' or 'Cold' weather).
2.  **Observations (O):** A set of observable symbols or outputs that are emitted from each state (e.g., 'number of ice creams eaten').
3.  **State Transition Probabilities (A):** The probability of moving from one hidden state to another. `Aᵢⱼ = P(state j at t+1 | state i at t)`.
4.  **Emission Probabilities (B):** The probability of observing a particular output while in a specific hidden state. `Bᵢ(k) = P(observation k | state i)`.
5.  **Initial State Probabilities (π):** The probability of the system starting in each hidden state.

**Working Principle:**
An HMM generates a sequence of observations by first choosing an initial hidden state according to **π**. Then, at each time step, it emits an observation according to the **emission probabilities (B)** of the current state and moves to a new hidden state according to the **transition probabilities (A)**.

**Three Fundamental Problems Solved by HMMs:**
1.  **Likelihood:** What is the probability of a given observation sequence? (Solved by the **Forward Algorithm**). This is used for classification.
2.  **Decoding:** What is the most likely sequence of hidden states that produced an observation sequence? (Solved by the **Viterbi Algorithm**).
3.  **Learning:** How can we adjust the model parameters (A, B, π) to best fit the observed data? (Solved by the **Baum-Welch Algorithm**).

---

### **Q. 3**

**(a) Explain the terms: (i) Autocorrelation (ii) Cross-correlation. (2 Marks)**

*   **(i) Autocorrelation:** A measure of the similarity between a signal and a time-delayed version of itself. It is used to find repeating patterns or periodicities within a single signal.
*   **(ii) Cross-correlation:** A measure of the similarity between two different signals as a function of the time lag applied to one of them. It is used to find the time delay between two signals or to detect a known signal within another signal.

**(b) Explain working of Non-negative Matrix Factorization. (3 Marks)**

Non-negative Matrix Factorization (NMF) is a dimensionality reduction technique that decomposes a high-dimensional non-negative matrix **V** into two smaller, non-negative matrices, **W** and **H**.
*   **Formula:** V ≈ WH
*   **Matrices:**
    *   **V:** The original data matrix (e.g., documents vs. words).
    *   **W:** The "basis" or "features" matrix (e.g., topics).
    *   **H:** The "coefficients" or "weights" matrix (e.g., how much of each topic is in each document).
*   **Working:** NMF uses an iterative optimization process to find W and H. The key constraint is that all elements must be non-negative, which leads to an additive, parts-based representation that is often easier for humans to interpret than other methods like PCA.

**(c) Explain Factor Analysis in detail. (7 Marks)**

Factor Analysis (FA) is a statistical and dimensionality reduction method used to identify the underlying latent structure among a set of observed, correlated variables. Its primary purpose is to represent these variables in terms of a smaller number of unobserved variables called **factors**.

**The Model:**
FA assumes that the observed variables (**X**) are linear combinations of unobserved common factors (**F**) plus a unique error term (**ε**).
*   **X = L * F + ε**
    *   **L (Factor Loadings):** A matrix of coefficients that represent the correlation between each observed variable and each factor. A high loading indicates a strong relationship.
    *   **F (Common Factors):** The latent variables that are believed to influence multiple observed variables and explain the correlations among them.
    *   **ε (Unique Factors/Error):** The portion of variance in each observed variable that is not explained by the common factors.

**Key Concepts:**
1.  **Communality:** The proportion of variance in an observed variable that is shared with other variables and explained by the common factors.
2.  **Factor Rotation:** A process applied after factor extraction to simplify the factor structure and make it more interpretable. Methods like **Varimax** rotate the factor axes to achieve a state where each variable loads highly on only one factor.

**Factor Analysis vs. PCA:**
While both are dimensionality reduction techniques, their goals differ:
*   **PCA** aims to find components that capture the maximum possible **total variance** in the data. It is a mathematical transformation.
*   **FA** aims to model the **shared variance** (covariance) among variables to uncover the underlying latent structure. It is a statistical model that separates shared and unique variance.

**(a) OR Define supervised and reinforcement learning techniques. (2 Marks)**

*   **Supervised Learning:** A machine learning approach where the algorithm learns from a labeled dataset, meaning each data point is tagged with a correct output or target. The goal is to learn a mapping function that can predict the output for new, unseen data. Examples include classification and regression.
*   **Reinforcement Learning:** A machine learning approach where an "agent" learns to make optimal decisions by interacting with an "environment". The agent receives "rewards" or "penalties" for its actions and learns a policy to maximize its cumulative reward over time through trial and error.

**(b) OR Write a short not on LDA. (3 Marks)**

Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique primarily used for classification problems.
*   **Goal:** To find a lower-dimensional space that maximizes the separability between classes.
*   **Method:** LDA projects the data onto a new axis (or axes) in a way that maximizes the distance between the means of the different classes while simultaneously minimizing the variance within each class.
*   **Versus PCA:** Unlike PCA (which is unsupervised and finds directions of maximum variance), LDA is supervised and finds directions that are optimal for discriminating between classes, making it highly effective as a pre-processing step for classification.

**(c) OR Explain working of PCA as dimensionality reduction method. (7 Marks)**

Principal Component Analysis (PCA) is a widely used unsupervised technique for dimensionality reduction. Its goal is to transform a dataset with many correlated variables into a smaller set of uncorrelated variables called **principal components**, while retaining most of the original information (variance).

**Working Steps:**
1.  **Standardize the Data:** Scale each feature to have a mean of 0 and a standard deviation of 1. This step is crucial to ensure that features with larger scales do not dominate the analysis.
2.  **Compute the Covariance Matrix:** Calculate the covariance matrix for the standardized dataset. This matrix describes the variance of each feature and the covariance between pairs of features.
3.  **Calculate Eigenvectors and Eigenvalues:** Decompose the covariance matrix to find its eigenvectors and eigenvalues.
    *   **Eigenvectors:** These represent the directions of the new axes (the principal components). They are orthogonal to each other.
    *   **Eigenvalues:** These indicate the amount of variance captured by each corresponding eigenvector. A higher eigenvalue means more variance.
4.  **Select Principal Components:** Sort the eigenvectors in descending order of their corresponding eigenvalues. The principal components are ranked by the amount of variance they explain. To reduce dimensionality from *n* to *k* dimensions, select the top *k* eigenvectors. The number *k* is often chosen to retain a certain percentage of the total variance (e.g., 95%).
5.  **Transform the Data:** Create a projection matrix using the selected *k* eigenvectors. Project the original standardized data onto this new, lower-dimensional subspace by taking the dot product of the data and the projection matrix. The result is the new, reduced dataset where each row represents the original data point in terms of the principal components.
