# Block Designs Enhancement Plan

**Date Created:** 2025-11-15
**Status:** Planning Complete - Ready for Implementation

---

## 🎯 **Proposed Block Designs Enhancement Plan**

### **Phase 1: UI Restructuring & Quick Wins** (Current Phase)

#### **1. Convert to Tabbed Interface** ✅ (User requested)
- Replace dropdown with tabs matching ANOVA/Factorial/Hypothesis Testing design
- Create 3 tabs: **RCBD**, **Latin Square**, **Graeco-Latin Square**
- Each tab shows only relevant parameters
- Better discoverability and consistency

#### **2. Display Relative Efficiency** ✅ (Backend already calculates!)
- **Status**: Calculated at line 255 in `block_designs.py` but NOT shown in UI
- Add efficiency metric card showing:
  - Relative Efficiency value (e.g., 1.45 = 45% more efficient than CRD)
  - Interpretation: "Using blocks is X% more efficient than a completely randomized design"
  - Visual indicator (badge/metric)

#### **3. Block Structure Visualization** ✅
- Create visual representation of the blocking structure:
  - Color-coded table showing treatment assignments by block
  - Highlight randomization pattern
  - Show run order if randomized

#### **4. Enhanced Block-Specific Diagnostics** ✅
- Add diagnostic panel specific to blocked designs:
  - **Block vs Treatment interaction plot** (check for interaction effects)
  - **Residuals by block** (check homogeneity across blocks)
  - **Normality test** (Shapiro-Wilk) for residuals
  - **Levene's test** for homogeneity of variance across blocks

#### **5. Design Information Panel** ✅
- Enhance the design info box with:
  - Design efficiency metrics
  - Degrees of freedom breakdown
  - Power of the design (based on replication)
  - Expected precision gain from blocking

---

### **What We WON'T Implement in Phase 1** (Future enhancements)

These are more complex and would require significant backend development:

❌ **Incomplete Block Designs** (BIB, PBIB, Youden) - Requires new statistical algorithms
❌ **Crossover Designs** - Clinical trial specific, different model
❌ **ANCOVA** - Requires covariate support
❌ **Missing Data Handling** - Complex statistical imputation

---

## 📊 **BLOCK DESIGNS AUDIT REPORT**

### CURRENT STATE SUMMARY

**Frontend File**: `/Users/nj/Desktop/MasterStat/frontend/src/pages/BlockDesigns.jsx` (590 lines)
**Backend File**: `/Users/nj/Desktop/MasterStat/backend/app/api/block_designs.py` (681 lines)

---

## 1. FRONTEND IMPLEMENTATION ANALYSIS

### UI Structure
- **Design Type Selection**: Uses a **dropdown** (not tabs) to select between three design types
- **Current Dropdown Options**:
  1. Randomized Complete Block Design (RCBD)
  2. Latin Square Design
  3. Graeco-Latin Square Design

### Design Parameters
- **RCBD**:
  - Number of Treatments (2-20)
  - Number of Blocks (2-20)
  - Randomize Run Order (checkbox)

- **Latin & Graeco-Latin**:
  - Square Size n×n (2-26 for Latin, 3-12 for Graeco-Latin)
  - Graeco-Latin excludes n=2 and n=6 (no orthogonal squares exist)

### Analysis Features Displayed

| Feature | Implemented | Details |
|---------|-------------|---------|
| Auto-design generation | YES | Generates design table with randomization |
| Data table entry | YES | Excel-like keyboard navigation with auto-fill |
| Response variable naming | YES | Customizable response name |
| Significance level (α) | YES | Default 0.05, user-configurable |
| Fixed vs Random blocks | YES | Checkbox for treating blocks as random effects (RCBD & Latin only) |
| Design info display | YES | Shows design type, total runs, treatments, blocks |

### Results Displayed (from ResultCard.jsx analysis)

**What IS shown:**
1. ANOVA Table with F, p-value, significance indicators
2. Treatment Means (with confidence intervals shown in backend)
3. Block Means
4. Grand Mean
5. Model R-squared (for fixed blocks)
6. Box plots by treatment
7. Residual diagnostic plots (Q-Q, residuals vs fitted, histogram)
8. Variance components (for random blocks): ICC, log-likelihood, AIC, BIC
9. Standard residuals and fitted values

**What is NOT shown/accessible:**
1. **Relative efficiency (calculated in backend, NOT displayed in UI)**
   - Backend calculates at line 180-188 in block_designs.py
   - Formula: `[(b-1)MS_block + b(t-1)MS_error] / [bt - 1)MS_error]`
   - Value is in result payload but not rendered by ResultCard

2. **Block means CI** - Only treatment means CI displayed
3. **Blockplot visualization** - No visual representation of blocking structure
4. **Residual analysis specific to blocks** - General residual plots shown, no block-specific diagnostics

---

## 2. BACKEND IMPLEMENTATION ANALYSIS

### API Endpoints Implemented

```
POST /api/block-designs/rcbd
POST /api/block-designs/latin-square
POST /api/block-designs/graeco-latin
POST /api/block-designs/generate/rcbd
POST /api/block-designs/generate/latin-square
POST /api/block-designs/generate/graeco-latin
```

### Design Types Supported

1. **RCBD (Randomized Complete Block Design)**
   - Fixed Blocks: OLS ANOVA with Type II SS
   - Random Blocks: Mixed Linear Model (MixedLM)
   - Supports 2-20 treatments × 2-20 blocks

2. **Latin Square**
   - Fixed Blocks: OLS ANOVA with row + column blocking
   - Random Blocks: MixedLM approximation
   - n×n designs where n ∈ [2, 26]

3. **Graeco-Latin Square**
   - Fixed Blocks only (no random effects option)
   - Two treatment factors + two blocking factors
   - n×n designs where n ∈ [3, 12], excluding n=2 and n=6

### Statistical Outputs Calculated

**For RCBD (Fixed Blocks):**
- Sum of squares (SS), degrees of freedom, F-statistic, p-value
- **Relative efficiency** ✓ (calculated but not exposed in UI)
- Residuals, fitted values, standardized residuals
- Treatment means with 95% CI
- Box plot data by treatment and block
- Model R-squared

**For RCBD (Random Blocks):**
- Variance components: block variance (σ²ᵦ), residual (σ²ₑ)
- Intraclass correlation (ICC = σ²ᵦ / (σ²ᵦ + σ²ₑ))
- Mixed model fit statistics: log-likelihood, AIC, BIC
- F-test for treatment effect

**For Latin Square (both fixed/random):**
- Row block means, column block means, treatment means
- Residuals and diagnostics (fixed blocks)
- Variance components (random blocks)

**For Graeco-Latin:**
- Latin treatment means
- Greek treatment means
- Row and column block means
- Model R-squared

### Efficiency Calculations

Backend implements **relative efficiency for RCBD (fixed blocks only)**:

```python
# Line 180-188
relative_efficiency = ((n_blocks - 1) * ms_block + n_blocks * (n_treatments - 1) * ms_error) /
                      ((n_blocks * n_treatments - 1) * ms_error)
```

**Status**: Calculated but **NOT displayed in frontend UI**

### Missing Backend Implementations

From the roadmap, these are completely absent:
1. **Incomplete Block Designs** (BIB, PBIB, Youden squares)
2. **Crossover Designs** (for clinical trials)
3. **Analysis of Covariance** (ANCOVA with blocking)
4. **Efficiency calculations for Graeco-Latin or other designs**
5. **Missing data handling** (imputation methods)
6. **Randomization visualization** (graphical blocking structure)

---

## 3. GAP ANALYSIS

### A. Features in Backend but NOT Exposed in Frontend

| Feature | Backend | Frontend | Impact |
|---------|---------|----------|--------|
| Relative Efficiency vs CRD | YES (RCBD) | NO | Users can't see design efficiency gains |
| Block Means CI | Partial (only means) | NO | Incomplete uncertainty quantification |
| Variance Components Detail | YES (random blocks) | YES | Well-implemented |
| Individual standardized residuals | YES | NO (only plotted) | Can't inspect specific points |
| Boxplot data | YES | YES | Good |
| Model fit statistics (AIC/BIC) | YES | YES | Good for random blocks |

### B. Completely Missing from Roadmap (Not Implemented Anywhere)

**High Priority (User-facing features):**
1. **Incomplete Block Designs (BIB, PBIB)**
   - Youden squares
   - Balance information matrices

2. **Crossover Designs**
   - Period effects
   - Carry-over effects

3. **Randomization Visualization**
   - Graphical representation of block structure
   - Treatment assignment visualization
   - Run order visualization

4. **Analysis of Covariance (ANCOVA)**
   - Covariate adjustment within blocks
   - Adjusted treatment means

**Medium Priority (Statistical rigor):**
5. **Missing Data Handling**
   - Imputation methods
   - Proper ANOVA with unbalanced data
   - Missing data diagnostics

6. **Enhanced Residual Analysis**
   - Block-specific diagnostics
   - Interaction plots (block × treatment)
   - Normality tests (Shapiro-Wilk for blocks)
   - Homogeneity tests across blocks

7. **Efficiency Calculations**
   - Relative efficiency display
   - Efficiency vs CRD comparison
   - Relative efficiency for Latin/Graeco-Latin

---

## 4. CURRENT UI STRUCTURE DETAILS

### Design Type Selection
```javascript
// Line 291-299 in BlockDesigns.jsx
<select value={designType} onChange={(e) => setDesignType(e.target.value)}>
  <option value="rcbd">Randomized Complete Block Design (RCBD)</option>
  <option value="latin">Latin Square Design</option>
  <option value="graeco">Graeco-Latin Square Design</option>
</select>
```

**This is a single dropdown, not tabs.** Each option completely changes the parameter inputs:
- RCBD: shows "Number of Treatments" and "Number of Blocks"
- Latin/Graeco: shows "Square Size (n × n)"

### Parameters Display Flow
1. Design Type dropdown (changes page layout)
2. Design-specific parameters
3. Response variable name
4. Randomization options (RCBD only)
5. Random blocks toggle (RCBD and Latin only)
6. Generated design info box
7. Data table (auto-populated)
8. Significance level
9. Analyze button

### Results Section
Uses generic `<ResultCard />` component which:
- Shows ANOVA table if `result.anova_table` exists
- Shows variance components if `result.variance_components` exists
- Shows boxplot data if `result.boxplot_data` exists
- Shows residual plots if residuals/fitted/standardized_residuals exist

**Notable**: ResultCard doesn't have specific handling for:
- `relative_efficiency` field (rendered only if explicitly checked in code, but not highlighted)
- Block-specific residual diagnostics
- Design efficiency comparison

---

## 5. WHAT WOULD UNLOCK QUICK WINS

### Low Effort, High Impact

1. **Display Relative Efficiency** (5 min frontend fix)
   - Backend already calculates it (line 255 in block_designs.py)
   - Just add to ResultCard rendering
   - Would show blocking value immediately

2. **Add Efficiency Explanation Panel** (15 min)
   - Calculate CRD MSE without blocking
   - Show comparison metric
   - Add interpretation text

3. **Create "Block Structure Visualization"** (30 min)
   - Use generated design data
   - Create simple table view of blocks
   - Color-code treatments per block

4. **Export Design Table as CSV** (20 min)
   - Convert tableData to CSV
   - Use existing export infrastructure

5. **Add Normality Test for Residuals** (30 min)
   - Integrate Shapiro-Wilk test
   - Backend: use scipy.stats.shapiro
   - Display p-value in ResultCard

### Medium Effort, Good Value

6. **Missing Data Handling** (2-3 hours)
   - Add "missing value" input option
   - Implement missing data flag
   - Use statsmodels unbalanced design handling

7. **ANCOVA Support** (3-4 hours)
   - Add covariate selection
   - Modify backend formula
   - Display adjusted means

8. **Incomplete Block Designs** (8-10 hours)
   - BIB design generation
   - PBIB analysis
   - Balance matrix generation

---

## 6. SUMMARY TABLE

| Category | Status | Count |
|----------|--------|-------|
| **Implemented & Displayed** | Complete | 5 features |
| **Implemented & Hidden** | Partial | 2 features (relative efficiency, individual residuals) |
| **Completely Missing** | None in Core | 7 major features from roadmap |
| **Design Types Supported** | 3/10 mentioned | RCBD, Latin, Graeco-Latin (no BIB, PBIB, crossover, etc.) |

---

## 7. DETAILED FEATURE CHECKLIST

### Roadmap Requirements

```
✅ RCBD support              - Implemented with fixed & random blocks
✅ Latin Square             - Implemented with fixed & random blocks
✅ Graeco-Latin Square      - Implemented (fixed blocks only)
✅ Auto-design generation   - Implemented with randomization
✅ Clean table interface    - Implemented with Excel-like navigation

❌ BIB, PBIB designs        - Not implemented
❌ Youden squares           - Not implemented
⚠️ Efficiency calculations  - Calculated but NOT displayed
❌ Randomization visualization - Not implemented (no blocking structure viz)
❌ Missing data handling    - Not implemented
❌ Crossover designs        - Not implemented
❌ ANCOVA                   - Not implemented
⚠️ Residual analysis        - Partially implemented (general, not block-specific)
```

---

## 📋 **Detailed Implementation Checklist**

### **Frontend Changes** (BlockDesigns.jsx)

- [ ] **Replace dropdown with 3-tab navigation**
  - [ ] Tab 1: RCBD (cyan theme)
  - [ ] Tab 2: Latin Square (purple theme)
  - [ ] Tab 3: Graeco-Latin Square (green theme)

- [ ] **Add Efficiency Display Component**
  - [ ] Show `relative_efficiency` from backend
  - [ ] Add comparison text: "X% more efficient than CRD"
  - [ ] Visual metric card with icon

- [ ] **Create BlockStructureVisualization component**
  - [ ] Color-coded treatment assignments
  - [ ] Block × Treatment matrix view
  - [ ] Randomization order display

- [ ] **Add BlockDiagnostics component**
  - [ ] Block-specific residual plots
  - [ ] Interaction plot (Block × Treatment)
  - [ ] Homogeneity tests display

- [ ] **Enhance Design Info Panel**
  - [ ] Show efficiency metrics
  - [ ] DF breakdown table
  - [ ] Design quality indicators

### **Backend Changes** (block_designs.py)

- [ ] **Add normality tests**
  - [ ] Shapiro-Wilk test for residuals
  - [ ] Return p-value and interpretation

- [ ] **Add homogeneity tests**
  - [ ] Levene's test across blocks
  - [ ] Return test statistic and p-value

- [ ] **Calculate block-treatment interaction data**
  - [ ] Means by block × treatment combinations
  - [ ] For interaction plot generation

- [ ] **Ensure relative_efficiency in all responses**
  - [ ] Currently only in RCBD fixed blocks
  - [ ] Add interpretation text

---

## 🎨 **Visual Mockup of Tab Structure**

```
┌─────────────────────────────────────────────────────────┐
│  Block Designs                                          │
│  Powerful experimental designs for controlling...       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  [RCBD]  [Latin Square]  [Graeco-Latin Square]         │
└─────────────────────────────────────────────────────────┘

[RCBD Tab Content]
┌─────────────────────────────────────────────────────────┐
│ Configuration                                           │
│ • Number of Treatments: [5]                            │
│ • Number of Blocks: [4]                                │
│ • Treat blocks as random: [✓]                          │
│ • Randomize run order: [✓]                             │
│                                                         │
│ Design Efficiency Metrics                               │
│ ┌─────────────────────────────────────────────┐        │
│ │ 📊 Relative Efficiency: 1.42                │        │
│ │ This design is 42% more efficient than CRD  │        │
│ └─────────────────────────────────────────────┘        │
│                                                         │
│ [Generated Design Table]                                │
│ [Results with Block-Specific Diagnostics]              │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ **Expected Impact**

| Enhancement | User Benefit | Implementation Time |
|-------------|--------------|---------------------|
| Tabbed UI | Better discoverability, consistent UX | 45 min |
| Efficiency Display | Understand blocking value | 20 min |
| Block Visualization | See treatment assignment pattern | 40 min |
| Block Diagnostics | Validate blocking assumptions | 60 min |
| Enhanced Info Panel | Better design understanding | 30 min |

**Total Estimated Time:** ~3 hours

---

## 📝 **Implementation Notes**

### Key Points to Remember:
1. Relative efficiency is already calculated in backend (line 255 of block_designs.py)
2. Design generation endpoints are separate from analysis endpoints
3. Random blocks use MixedLM, fixed blocks use OLS
4. Graeco-Latin doesn't support random blocks (limitation of the design)

### Files to Modify:
- `frontend/src/pages/BlockDesigns.jsx` - Main UI restructuring
- `backend/app/api/block_designs.py` - Add diagnostic tests
- Potentially create new components:
  - `frontend/src/components/BlockStructureViz.jsx`
  - `frontend/src/components/BlockDiagnostics.jsx`
  - `frontend/src/components/EfficiencyMetric.jsx`

---

**Status:** Ready for Implementation
**Priority:** Phase 1 - High Priority Quick Wins
**Estimated Completion:** 3 hours
