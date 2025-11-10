# Mixed Models Section - Development Roadmap

## Completed ✅

1. **Mixed ANOVA** - Between and within subjects factors
2. **Split-Plot Design** - Whole plot and subplot factors with visualizations
3. **Nested Design** - Hierarchical structures with variance components

## Pending Implementation

### 1. Repeated Measures ANOVA (IN PROGRESS)
- Within-subjects design where each participant is measured multiple times
- Essential for time-series experiments, pre-post designs
- **Statistical Features:**
  - Sphericity testing (Mauchly's test)
  - Corrections (Greenhouse-Geisser, Huynh-Feldt)
  - Within-subjects effects and contrasts
- **Visualizations:**
  - Profile plots (time trends)
  - Within-subject variability plots
  - Individual trajectories
  - Effect size plots

### 2. Crossed Random Effects
- Both factors are random and crossed (not nested)
- Example: Students × Tests where all students take all tests
- Different from nested where levels of B are unique to each A
- **Statistical Features:**
  - Variance components for both factors
  - Interaction variance
  - Proper F-tests for crossed designs
- **Visualizations:**
  - Variance component breakdown
  - Interaction plots
  - Random effects distributions

### 3. General Linear Mixed Models (LMM)
- More flexible than ANOVA-based approaches
- Maximum likelihood or REML estimation
- Random intercepts and slopes
- Useful for unbalanced designs and missing data
- **Statistical Features:**
  - Fixed and random effects
  - Covariance structures
  - Model comparison (AIC, BIC, likelihood ratio tests)
- **Visualizations:**
  - Random effects plots
  - Fitted vs. observed
  - Residual diagnostics for mixed models

### 4. Repeated Measures with Between-Subjects Factors
- Combines repeated measures with grouping factors
- Example: Treatment groups measured over time
- **Statistical Features:**
  - Between-subjects and within-subjects effects
  - Interactions between factors
  - Sphericity corrections
- **Visualizations:**
  - Profile plots by group
  - Interaction plots
  - Time × Group effects

## Notes
- After Repeated Measures ANOVA, return to the main development roadmap
- Each implementation should include comprehensive visualizations
- All analyses should support data export and PNG chart export
