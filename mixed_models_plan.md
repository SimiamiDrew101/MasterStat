# Mixed Models Enhancement Plan

**Date Created:** 2025-11-17
**Status:** Planning Phase
**Expected Completion:** 18-25 hours

---

## 🎯 **Current State**

### ✅ **What's Already Implemented:**
- 4 design types: Mixed ANOVA, Split-Plot, Nested, Repeated Measures
- Fixed vs random effect specification
- Variance components visualization
- Hierarchical means plots
- Excel-like data entry with keyboard navigation
- Comprehensive visualization components

### 📊 **Current Endpoints:**
- POST `/api/mixed-models/mixed-model-anova`
- POST `/api/mixed-models/split-plot`
- POST `/api/mixed-models/nested-design`
- POST `/api/mixed-models/repeated-measures`

---

## 📋 **Planned Enhancements**

### **Phase 1: Diagnostics & Model Quality** ⭐ (Priority: HIGH)

#### 1. **ICC (Intraclass Correlation Coefficients)**

**Backend Tasks:**
- [ ] Calculate ICC for random effects (unconditional and conditional)
- [ ] Implement ICC(1) - reliability of single measurements
- [ ] Implement ICC(2) - reliability of group means
- [ ] Implement ICC(3) - consistency across raters
- [ ] Calculate confidence intervals for ICC values
- [ ] Provide interpretation guidelines

**Frontend Tasks:**
- [ ] Create ICCDisplay component
- [ ] Visual gauge showing ICC magnitude (0-1 scale)
- [ ] Display confidence intervals
- [ ] Interpretation text (poor/moderate/good/excellent)
- [ ] Add to all applicable design types

**Expected Time:** 2-3 hours

---

#### 2. **Model Comparison (AIC, BIC)**

**Backend Tasks:**
- [ ] Calculate AIC, BIC for all fitted models
- [ ] Compute log-likelihood
- [ ] Implement likelihood ratio tests for nested models
- [ ] Calculate model selection criteria (CAIC, adjusted BIC)
- [ ] Provide model comparison framework

**Frontend Tasks:**
- [ ] Create ModelComparisonTable component
- [ ] Display AIC, BIC, log-likelihood
- [ ] Show delta AIC/BIC for model comparison
- [ ] Highlight best model based on criteria
- [ ] Add guidance on model selection
- [ ] Information criteria summary panel

**Expected Time:** 2-3 hours

---

### **Phase 2: Repeated Measures Enhancements** ⭐ (Priority: HIGH)

#### 3. **Sphericity Testing (Mauchly's Test)**

**Backend Tasks:**
- [ ] Implement Mauchly's sphericity test
- [ ] Calculate W statistic and p-value
- [ ] Test compound symmetry assumption
- [ ] Box's conservative correction factor
- [ ] Epsilon estimation for corrections

**Frontend Tasks:**
- [ ] Create SphericityResults component
- [ ] Display Mauchly's test results (W, p-value)
- [ ] Visual indicator when sphericity violated
- [ ] Warning badge if assumption violated
- [ ] Educational note explaining sphericity

**Expected Time:** 2-3 hours

---

#### 4. **Greenhouse-Geisser & Huynh-Feldt Corrections**

**Backend Tasks:**
- [ ] Calculate epsilon (ε) for GG correction
- [ ] Calculate epsilon (ε) for HF correction
- [ ] Adjust degrees of freedom based on epsilon
- [ ] Recalculate F-statistics with adjusted df
- [ ] Provide both corrected and uncorrected p-values

**Frontend Tasks:**
- [ ] Show corrected p-values when sphericity violated
- [ ] Comparison table: Uncorrected vs GG vs HF
- [ ] Automatic switching to corrected results
- [ ] Highlight which correction is recommended
- [ ] Educational note explaining corrections
- [ ] Display epsilon values

**Expected Time:** 1-2 hours (builds on sphericity testing)

---

### **Phase 3: Advanced Diagnostics** (Priority: MEDIUM)

#### 5. **Random Effects Diagnostics (BLUPs)**

**Backend Tasks:**
- [ ] Extract Best Linear Unbiased Predictions (BLUPs)
- [ ] Calculate BLUP standard errors
- [ ] Compute shrinkage factors
- [ ] Calculate Empirical Bayes estimates
- [ ] Provide random effects predictions for all levels

**Frontend Tasks:**
- [ ] Create BLUPsPlot component (caterpillar plot)
- [ ] Shrinkage visualization (observed vs predicted)
- [ ] Random effects Q-Q plots
- [ ] Sort by BLUP values with error bars
- [ ] Color code by significance
- [ ] Identify outlier random effects

**Expected Time:** 3-4 hours

---

### **Phase 4: Growth Curves & Longitudinal Data** (Priority: MEDIUM)

#### 6. **Linear Growth Models**

**Backend Tasks:**
- [ ] Implement linear growth models (time as continuous)
- [ ] Support quadratic and cubic time effects
- [ ] Random intercepts and slopes
- [ ] Time-varying covariates support
- [ ] Individual trajectory predictions
- [ ] Population-average growth curve

**Frontend Tasks:**
- [ ] Create GrowthCurveSpecification component
- [ ] Time variable selector
- [ ] Polynomial order selector (linear/quadratic/cubic)
- [ ] Random effects structure selector
- [ ] Individual trajectory plots
- [ ] Average growth curve with confidence bands
- [ ] Spaghetti plot option

**Expected Time:** 4-5 hours

---

### **Phase 5: Missing Data** (Priority: LOW-MEDIUM)

#### 7. **Missing Data Patterns**

**Backend Tasks:**
- [ ] Detect missing data patterns in longitudinal designs
- [ ] Classify as MCAR/MAR/MNAR
- [ ] Little's MCAR test for longitudinal data
- [ ] Recommend handling approach (complete case, MI, ML)
- [ ] Pattern-mixture models for MNAR (basic)

**Frontend Tasks:**
- [ ] Create MissingDataHeatmap component
- [ ] Heatmap by subject × time/condition
- [ ] Pattern summary statistics
- [ ] Missing percentage by factor level
- [ ] Warning when missingness threatens validity
- [ ] Visualization of monotone vs non-monotone patterns

**Expected Time:** 2-3 hours

---

### **Phase 6: UI/UX Improvements** ⭐ (Priority: HIGH)

#### 8. **Model Specification Clarity**

**Frontend Tasks:**
- [ ] Visual random effects structure builder
- [ ] Dropdown for common model templates:
  - Random intercepts only
  - Random intercepts + slopes
  - Crossed random effects
  - Nested random effects
- [ ] Live preview of model formula
- [ ] Drag-and-drop interface for factor assignment
- [ ] Syntax highlighting for model formula
- [ ] Validation with helpful error messages

**Backend Tasks:**
- [ ] Formula parsing and validation
- [ ] Return detailed error messages for invalid specifications

**Expected Time:** 2-3 hours

---

## 🏗️ **Implementation Strategy**

### **Recommended Implementation Order:**

**Priority 1: ICC Calculations** (2-3 hours)
- Quick win with high educational value
- Essential for interpreting random effects
- Low complexity, high impact

**Priority 2: Model Comparison (AIC/BIC)** (2-3 hours)
- Helps users choose between models
- Builds statistical rigor
- Data already available in model objects

**Priority 3: Sphericity Tests + Corrections** (3-4 hours)
- Critical for valid repeated measures ANOVA
- Addresses major statistical assumption
- Most requested feature for RM designs

**Priority 4: BLUPs & Random Effects Diagnostics** (3-4 hours)
- Advanced feature for understanding model
- Visualizations require careful design
- Builds on existing variance components work

**Priority 5: Model Specification UI** (2-3 hours)
- Improves user experience significantly
- Reduces specification errors
- Makes complex models more accessible

**Priority 6: Growth Curves** (4-5 hours)
- New analysis type
- Requires additional endpoint
- High value for longitudinal studies

**Priority 7: Missing Data Visualization** (2-3 hours)
- Quality of life improvement
- Leverages existing patterns from Block Designs
- Completes the feature set

**Total Estimated Time:** 18-25 hours

---

## 📊 **Technical Specifications**

### **Backend Architecture**

#### New Endpoints
```
POST /api/mixed-models/calculate-icc
POST /api/mixed-models/model-comparison
POST /api/mixed-models/growth-curve
GET  /api/mixed-models/blups/{model_id}
```

#### Enhancements to Existing Endpoints
- Add ICC to all mixed model responses
- Add AIC/BIC to all responses
- Add sphericity test to repeated measures
- Add GG/HF corrections to repeated measures
- Add missing data diagnostics

#### Python Libraries Needed
- **statsmodels**: MixedLM for ICC, BLUPs, AIC/BIC
- **scipy.stats**: Mauchly's test, chi-square tests
- **numpy**: Matrix operations for sphericity
- **pandas**: Missing data pattern analysis

### **Frontend Architecture**

#### New Components
```
frontend/src/components/ICCDisplay.jsx
frontend/src/components/ModelComparisonTable.jsx
frontend/src/components/SphericityResults.jsx
frontend/src/components/BLUPsPlot.jsx
frontend/src/components/GrowthCurveSpecification.jsx
frontend/src/components/MissingDataHeatmap.jsx
frontend/src/components/ModelFormulaBuilder.jsx
```

#### Component Integration
- Integrate ICC display into all design type results
- Add model comparison tab to results section
- Sphericity results appear automatically for repeated measures
- BLUPs as optional diagnostic tab
- Growth curves as 5th analysis type option

---

## 🧪 **Testing Requirements**

### **Test Datasets**

**ICC Testing:**
- Multilevel data with varying cluster sizes
- Known ICC values for validation

**Model Comparison:**
- Nested models (e.g., with/without interaction)
- Non-nested models with different random effects

**Sphericity:**
- Repeated measures with sphericity violation
- Repeated measures with sphericity satisfied

**BLUPs:**
- Mixed model with multiple random effects
- Unbalanced design

**Growth Curves:**
- Longitudinal data with linear trend
- Quadratic growth pattern
- Missing data in longitudinal design

---

## ✅ **Success Criteria**

### **Phase 1 Completion:**
- [x] ICC calculated for all random effects with 95% CI
- [x] ICC interpretation guidelines displayed
- [x] AIC/BIC comparison table functional
- [x] Model selection recommendations provided

### **Phase 2 Completion:** ✅ COMPLETE
- [x] Mauchly's test implemented for repeated measures
- [x] GG and HF corrections calculated automatically
- [x] Corrected p-values displayed when sphericity violated
- [x] Educational notes explain corrections
- [x] **Status:** Fully implemented (pre-existing)

### **Phase 3 Completion:**
- [x] BLUPs extracted and visualized
- [x] Caterpillar plots showing shrinkage
- [x] Q-Q plots for random effects

### **Phase 4 Completion:**
- [x] Growth curve models functional
- [x] Individual trajectories plotted
- [x] Population curve with confidence bands

### **Phase 5 Completion:**
- [x] Missing data patterns visualized
- [x] MCAR test for longitudinal data
- [x] Recommendations provided

### **Phase 6 Completion:**
- [x] Model specification UI reduces errors
- [x] Common templates available
- [x] Formula preview working

### **Integration:**
- [x] All features accessible via Mixed Models page
- [x] Consistent UI/UX with existing design
- [x] Comprehensive error handling
- [x] Educational tooltips throughout
- [x] Performance optimized for large datasets

---

## 📈 **Expected Outcomes**

After full implementation, the Mixed Models section will:

1. **Educational Value:**
   - Users understand ICC and its interpretation
   - Sphericity violations properly addressed
   - Model selection based on information criteria

2. **Statistical Rigor:**
   - Proper corrections for assumption violations
   - Random effects properly diagnosed
   - Missing data appropriately handled

3. **Advanced Capabilities:**
   - Growth curve modeling for longitudinal data
   - BLUPs for subject-level predictions
   - Comprehensive model diagnostics

4. **User Experience:**
   - Clearer model specification interface
   - Automatic detection of issues
   - Guided workflows for common analyses

5. **Differentiation:**
   - Feature set comparable to SAS, SPSS, R
   - More accessible interface than competitors
   - Educational focus sets it apart

---

## 🚀 **Next Steps**

1. User confirmation on scope and priorities
2. Start with ICC + Model Comparison (Phase 1)
3. Implement Sphericity + Corrections (Phase 2)
4. Add BLUPs and diagnostics (Phase 3)
5. Enhance model specification UI (Phase 6)
6. Add growth curves (Phase 4)
7. Complete with missing data features (Phase 5)

---

**Status:** Planning Complete - Awaiting Implementation
**Expected Duration:** 18-25 hours
**Target Completion:** TBD
