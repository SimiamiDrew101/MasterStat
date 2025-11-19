# Block Designs Phase 2 Implementation Plan

**Date Created:** 2025-11-15
**Status:** Planning Phase
**Expected Completion:** 8-12 hours

---

## 🎯 **Phase 2 Objectives**

Implement four major advanced features for block designs:

1. **Incomplete Block Designs (BIB, PBIB, Youden)**
2. **Crossover Designs**
3. **ANCOVA (Analysis of Covariance)**
4. **Missing Data Handling**

---

## 📋 **Feature 1: Incomplete Block Designs**

### Overview
When complete blocks are not feasible (block size < number of treatments), incomplete block designs allow efficient experimentation with partial blocks.

### Types to Implement

#### 1.1 Balanced Incomplete Block Design (BIB)
**Requirements:**
- Every treatment appears in the same number of blocks (r)
- Every block contains the same number of treatments (k)
- Every pair of treatments appears together in the same number of blocks (λ)

**Constraints:**
- λ(v-1) = r(k-1) where v = number of treatments
- bk = vr where b = number of blocks

**Backend Tasks:**
- [ ] Generate BIB designs for common parameter sets
- [ ] Implement intrablock analysis
- [ ] Calculate adjusted treatment means
- [ ] Compute efficiency factor (EF)
- [ ] Validate BIB existence conditions

**Frontend Tasks:**
- [ ] Add BIB tab with parameter inputs (v, b, k, r, λ)
- [ ] Design generation interface
- [ ] Display concurrence matrix
- [ ] Show efficiency factor and interpretation

#### 1.2 Partially Balanced Incomplete Block Design (PBIB)
**Requirements:**
- Two associate classes
- Treatments in same associate class appear together with same frequency

**Backend Tasks:**
- [ ] Implement PBIB-2 designs
- [ ] Calculate association scheme
- [ ] Compute adjusted treatment means
- [ ] Handle missing pairs

**Frontend Tasks:**
- [ ] PBIB design selection interface
- [ ] Association matrix visualization
- [ ] Treatment relationship display

#### 1.3 Youden Square
**Requirements:**
- Incomplete Latin Square (fewer rows than columns)
- Each treatment appears once per column
- Balanced row occurrences

**Backend Tasks:**
- [ ] Generate Youden squares for common sizes
- [ ] Row and column analysis
- [ ] Efficiency calculations

**Frontend Tasks:**
- [ ] Youden square parameter selection
- [ ] Grid visualization similar to Latin Square
- [ ] Efficiency comparison with Latin Square

---

## 📋 **Feature 2: Crossover Designs**

### Overview
Used in clinical trials where subjects receive multiple treatments in sequence. Critical for controlling subject variability and studying period effects.

### Types to Implement

#### 2.1 Two-Treatment Two-Period (2×2) Crossover
**Requirements:**
- Two sequences: AB and BA
- Analysis of period effects and carryover effects
- Baseline measurements optional

**Backend Tasks:**
- [ ] Implement mixed model for crossover
- [ ] Test for period effect
- [ ] Test for treatment × period interaction
- [ ] Test for carryover (first-order)
- [ ] Calculate washout period recommendations

**Frontend Tasks:**
- [ ] Crossover design tab
- [ ] Sequence assignment display (AB vs BA)
- [ ] Period effect visualization
- [ ] Carryover effect interpretation

#### 2.2 Multi-Period Crossover Designs
**Common Designs:**
- Latin Square crossover (3+ treatments, 3+ periods)
- Williams designs (balanced for carryover)
- Balanced crossover (all treatment sequences)

**Backend Tasks:**
- [ ] Generate Williams designs
- [ ] Balanced sequence generation
- [ ] Multi-period mixed models
- [ ] Carryover contrast estimation

**Frontend Tasks:**
- [ ] Multi-period sequence visualization
- [ ] Timeline plot for each subject
- [ ] Carryover effect matrix

---

## 📋 **Feature 3: ANCOVA (Analysis of Covariance)**

### Overview
Adjust for continuous covariates that affect the response but are not of primary interest. Increases precision by removing covariate variation.

### Implementation Requirements

**Backend Tasks:**
- [ ] Modify RCBD analysis to accept covariates
- [ ] Implement ANCOVA model: Y = μ + block + treatment + β(X - X̄) + ε
- [ ] Test homogeneity of regression slopes
- [ ] Calculate adjusted treatment means
- [ ] Compute adjusted standard errors
- [ ] Perform adjusted contrasts and post-hoc tests

**Frontend Tasks:**
- [ ] Add covariate column selection from data table
- [ ] Toggle for "Include Covariate Analysis"
- [ ] Display homogeneity of slopes test results
- [ ] Show adjusted vs unadjusted means comparison
- [ ] Covariate effect visualization (regression plot)
- [ ] Adjusted confidence intervals display

**Additional Features:**
- Multiple covariate support
- Covariate × treatment interaction test
- Visual diagnostics for covariate model

---

## 📋 **Feature 4: Missing Data Handling**

### Overview
Handle missing observations in block designs without discarding entire blocks or treatments. Critical for real-world experiments.

### Implementation Requirements

**Backend Tasks:**
- [ ] Detect missing data patterns
- [ ] Implement missing data methods:
  - [ ] Listwise deletion (complete case analysis)
  - [ ] Mean imputation (simple)
  - [ ] EM algorithm imputation
  - [ ] Multiple imputation (basic)
- [ ] Adjust degrees of freedom for missing data
- [ ] Calculate unbalanced ANOVA
- [ ] Provide missing data diagnostics:
  - [ ] Missing data pattern analysis
  - [ ] MCAR/MAR/MNAR assessment
  - [ ] Little's MCAR test
- [ ] Handle missing data in efficiency calculations

**Frontend Tasks:**
- [ ] Visual indicator for missing cells in data table
- [ ] Allow empty cells or "NA" input
- [ ] Missing data summary panel
- [ ] Missing data pattern heatmap
- [ ] Imputation method selector
- [ ] Display imputed values (with visual distinction)
- [ ] Comparison of complete case vs imputed analysis
- [ ] Warning messages for high missingness

**Advanced Features:**
- [ ] Sensitivity analysis (compare imputation methods)
- [ ] Pattern-mixture models for MNAR
- [ ] Recommended imputation method based on pattern

---

## 🏗️ **Implementation Strategy**

### Order of Implementation

**Priority 1: ANCOVA** (2-3 hours)
- Least complex algorithmically
- Builds on existing RCBD infrastructure
- High user value (precision improvement)

**Priority 2: Missing Data Handling** (3-4 hours)
- Essential for real-world data
- Multiple imputation methods needed
- Diagnostic visualizations

**Priority 3: Crossover Designs** (2-3 hours)
- New design class (separate from blocking)
- Clinical trial focus
- Moderate complexity

**Priority 4: Incomplete Block Designs** (3-4 hours)
- Most algorithmically complex
- Requires design generation algorithms
- BIB, PBIB, and Youden each need separate implementations

**Total Estimated Time:** 10-14 hours

---

## 📊 **Technical Specifications**

### Backend Architecture

#### New Endpoints
```
POST /api/block-designs/bib
POST /api/block-designs/pbib
POST /api/block-designs/youden
POST /api/block-designs/crossover-2x2
POST /api/block-designs/crossover-williams
POST /api/block-designs/rcbd-ancova
POST /api/block-designs/impute-missing

GET /api/block-designs/generate/bib
GET /api/block-designs/generate/youden
GET /api/block-designs/generate/williams
```

#### Python Libraries Needed
- **statsmodels**: ANCOVA, mixed models, imputation
- **scipy.stats**: Additional tests
- **pandas**: Missing data handling
- **numpy**: Matrix operations for incomplete designs
- **itertools**: Design generation (combinations, permutations)

### Frontend Architecture

#### New Components
```
frontend/src/components/IncompleteBlockDesigns.jsx
frontend/src/components/CrossoverDesignSelector.jsx
frontend/src/components/CovariateSelector.jsx
frontend/src/components/MissingDataPanel.jsx
frontend/src/components/ImputationMethodSelector.jsx
frontend/src/components/EfficiencyComparison.jsx
```

#### UI Integration
- Add 3 new tabs to Block Designs:
  - "Incomplete Blocks" (BIB, PBIB, Youden sub-tabs)
  - "Crossover Designs"
  - "Advanced Options" (ANCOVA toggle, Missing Data settings)

---

## 🧪 **Testing Requirements**

### Test Datasets

**BIB Design:**
- v=7, b=7, r=3, k=3, λ=1 (classic BIB)
- Agricultural yield data

**Crossover Design:**
- 2×2 crossover with 20 subjects
- Clinical trial blood pressure data

**ANCOVA:**
- RCBD with baseline covariate
- Adjusted treatment means validation

**Missing Data:**
- RCBD with 10% random missing (MCAR)
- RCBD with 20% treatment-related missing (MAR)

---

## 📝 **Documentation Requirements**

Each feature needs:
- [ ] User guide explaining when to use the design
- [ ] Statistical background and assumptions
- [ ] Interpretation guidelines
- [ ] Example datasets
- [ ] Common pitfalls and warnings

---

## ✅ **Success Criteria**

### Phase 2 Completion Checklist

**Incomplete Block Designs:**
- [ ] Generate at least 3 common BIB designs
- [ ] Analyze BIB with intrablock analysis
- [ ] Calculate and display efficiency factor
- [ ] Implement at least one Youden square

**Crossover Designs:**
- [ ] 2×2 crossover with period and carryover tests
- [ ] Williams design generation (3×3, 4×4)
- [ ] Period effect visualization

**ANCOVA:**
- [ ] Single covariate ANCOVA for RCBD
- [ ] Homogeneity of slopes test
- [ ] Adjusted treatment means display
- [ ] Covariate effect plot

**Missing Data:**
- [ ] Detect and report missing patterns
- [ ] Implement mean imputation
- [ ] Implement EM imputation
- [ ] Missing data diagnostics display
- [ ] Unbalanced ANOVA calculation

**Integration:**
- [ ] All features accessible via Block Designs page
- [ ] Consistent UI/UX with Phase 1
- [ ] Comprehensive error handling
- [ ] Educational tooltips throughout

---

## 🚀 **Next Steps**

1. Get user confirmation on scope and priorities
2. Start with ANCOVA (quick win, builds confidence)
3. Implement Missing Data Handling (high practical value)
4. Add Crossover Designs (new design class)
5. Finish with Incomplete Blocks (most complex)

---

**Status:** Awaiting Implementation
**Expected Duration:** 10-14 hours
**Target Completion:** TBD
