# Peer Review — Round 1
## Manuscript: "Navigating the Polycrisis: Institutional Responses and Sustainable Cluster Resilience in China's Traditional Textile Industry"
## Journal: *Sustainability* (MDPI)

---

## Overall Assessment: MAJOR REVISION

This manuscript addresses a timely and important topic: how local government policy interventions shape the resilience of traditional industrial clusters under compounding shocks. The Gaoyang towel cluster is an analytically rich case, and the mixed-methods design combining NLP with quasi-experimental causal inference is well-motivated. The conceptual distinction between "baseline resilience" and "value resilience" is novel and potentially generative. The honest reporting of non-significant results is commendable.

However, the manuscript has several significant issues that must be addressed before it can be accepted. These fall into three categories: (A) methodological transparency and alignment between claims and evidence, (B) data documentation gaps, and (C) structural and formatting issues for the target journal.

---

## A. MAJOR ISSUES (Must Address)

### A1. Method Description: BERTopic vs. LLM Scoring

**[HIGH]** The manuscript describes using "Dynamic BERTopic" (Grootendorst, 2022) for policy topic extraction. However, BERTopic is an unsupervised topic modeling technique — it discovers latent topics from document collections without human annotation. The manuscript's policy attention metrics (environment, e-commerce, brand) appear to require supervised classification into predefined policy categories. This creates a tension: 

- If the manuscript used **unsupervised BERTopic**: The topic labels (environment, e-commerce, brand) would emerge from the data. The paper needs to show the actual topic keywords, coherence scores, and the process by which topics were interpreted as "environmental" or "e-commerce."
- If the manuscript used **supervised LLM scoring** (as suggested by the reference to Hansen 2026's LLM annotation validation framework): The method section should describe the LLM, prompt design, and scoring rubric, not BERTopic.

**Required:** Clarify exactly what method was used. If BERTopic, report topic keywords, coherence scores, and inter-topic distance maps. If LLM scoring, describe the model (version, parameters), prompt engineering, and validation protocol. The term "Dynamic BERTopic" needs clarification — standard BERTopic is static; describe any temporal modeling explicitly.

### A2. The 25-Year Data Constraint

**[HIGH]** The manuscript describes analyzing "25 consecutive Gaoyang County government work reports (2000–2024, ca. 200,000 Chinese characters)." Prior diagnostic work on this project revealed that pre-2015 reports were summary versions (<1KB), potentially too short for reliable NLP analysis. This constraint has fundamental implications:

- If early reports are summaries, their topic distributions may reflect **document length artifacts** rather than genuine policy attention shifts
- BERTopic embeddings from very short texts (<1,000 characters) may be unreliable
- This could explain why Phase 1 (2000–2008) appears dominated by "scale expansion" — short summary reports may only mention the most prominent topics

**Required:** (a) Report the character/word count distribution across the 25 reports. (b) If certain years have substantially shorter reports, discuss how this affects topic modeling reliability. (c) Consider a robustness check excluding or separately analyzing pre-2015 years. (d) At minimum, transparently disclose this constraint.

### A3. SCM Donor Pool Validity

**[HIGH]** The manuscript honestly notes that "the synthetic weight is concentrated on a single donor county, indicating limited donor pool similarity." This is a serious concern for SCM validity. When the synthetic control degenerates to a single unit, it essentially becomes a case comparison rather than a properly synthesized counterfactual. The pre-treatment RMSPE and the ratio of post/pre RMSPE should be reported. Furthermore:

- The manuscript reports placebo tests ranking Gaoyang 6th of 23 counties (top 30%, below 5% threshold) — this should be stated explicitly in the main text, not just the LaTeX version
- The post-2022 reversal (gap from +186.8 to −36.0) suggests either treatment effect decay or poor pre-treatment fit propagating forward
- With only 23 donor counties and pre-treatment fit concentrated on one county, the SCM evidence should be characterized as "exploratory" rather than "causal"

**Required:** Report RMSPE values. Discuss the implications of single-donor concentration for causal interpretation. Move the placebo test ranking to the main text.

### A4. ITS Sample Size and N Consistency

**[MODERATE]** Table 1 reports N=25 for key variables, but the ITS specification states N=27 (2000–2024). This discrepancy needs resolution. Furthermore, with N=25–27 for a single time series:

- Newey-West HAC standard errors with lag=1 are questionable — the rule of thumb for HAC is $0.75 \times T^{1/3}$, which would suggest lag=2 for T=25
- The statistical power to detect anything short of very large effects is minimal
- The non-significant results (p=0.10–0.15) should be interpreted in the context of this power constraint

**Required:** Reconcile N across tables. Justify HAC lag choice. Add a brief power analysis or discussion of minimum detectable effect size.

### A5. NLP Construct Validity: Missing Key Diagnostic

**[HIGH—but fixable]** The manuscript cites Hansen (2026) for "Annotation Backtranslation" as a validity check, stating "the model successfully reconstructed original policy semantics from extracted scores, proving the objective purity of our metrics." This is an extremely strong claim ("proving... objective purity") that is unlikely to be justified by any single validation technique. Furthermore, prior diagnostic work on this project revealed:

- That raw policy scores correlated r=0.959 with document length (number of paragraphs/chunks)
- That LLM scores and official policy support indices showed weak-to-negative correlation before residualization

If BERTopic was used instead of LLM scoring, these specific diagnostics may not apply. But the general concern remains: **NLP-derived policy metrics must be validated against document length artifacts and external benchmarks.**

**Required:** (a) Report the correlation between topic prevalence and document length for each topic. (b) If an external policy index is available (the manuscript mentions the Hebei-Gaoyang Textile Index), report the correlation between BERTopic-derived policy attention and that external benchmark. (c) Tone down the "objective purity" language — no single validation technique "proves" construct validity.

---

## B. MODERATE ISSUES (Should Address)

### B1. Theoretical Framework — Missing Micro-Foundations

The conceptual framework (baseline resilience → value resilience) is compelling but underdeveloped. Specifically:

- The compliance cost socialization mechanism would benefit from a simple formalization (e.g., how centralized wastewater treatment changes the cost function for a representative SME)
- The transaction cost reduction mechanism for e-commerce/branding is asserted rather than derived
- H3 (structural heterogeneity) is stated but receives only descriptive treatment — this is fine, but the hypothesis should be framed more tentatively

**Suggested:** Add 1–2 paragraphs sketching the microeconomic logic more formally. A simple diagram showing how public goods provision shifts the cost curves for SMEs would strengthen the framework considerably.

### B2. Reference List — Below Journal Recommendations

The manuscript has 22 references. MDPI *Sustainability* recommends 25+. Additionally:

- Several foundational texts are cited in older editions (e.g., Boschma 2015 rather than more recent review pieces)
- The text-as-data methodology literature is thin — Grimmer & Stewart (2013, *Political Analysis*) and Gentzkow, Kelly & Taddy (2019, *Journal of Economic Literature*) are standard references for NLP-based policy measurement and should be included
- Grey literature citation: Hansen (2026) is a Fed working paper — verify it is publicly accessible and provide the DOI/URL

**Required:** Add at least 5–8 references. Prioritize: (a) text-as-data methodology, (b) recent (2022–2026) resilience empirics, (c) Chinese-language policy evaluation studies.

### B3. Mechanism Tests — Statistical Power Concern

Table 5 reports mechanism regressions with N=25 and R²=0.44/0.32. With effectively 2–3 predictors and 25 observations, the degrees of freedom are tight. The price index analysis covers only 2020–2026 (N≈25 quarterly observations), which the manuscript acknowledges — but the N in Table 5 says 25, which is confusing if these are annual observations.

**Required:** Clarify the observation count and frequency for mechanism regressions. Report adjusted R². Consider reporting confidence intervals rather than (or in addition to) p-values, given the small N.

### B4. Figures Referenced but Not Described

Figures 1–3 are referenced but their captions are minimal. For example:
- Figure 2: "SDiD Complex" is mentioned in the markdown but the paper only uses SCM, not SDiD
- Figure 3: The "Policy Synergy Heatmap showing high synergy (0.88) between E-commerce and Regional Branding" is described in a caption but this synergy analysis is not described in the methods or results sections

**Required:** Ensure figure captions accurately describe what is shown. If SDiD was not used, the figure title should not reference it. Add a methods subsection describing the synergy/heatmap analysis if it is to be presented.

---

## C. MINOR ISSUES & FORMATTING

### C1. Abstract Length
The abstract is approximately 250+ words. *Sustainability* guidelines specify ~200 words maximum. Trim to essentials.

### C2. Author Contributions
"Conceptualization, Y.S. and Y.S." — both authors have the same initials. Use full names or distinguish with given name initials (e.g., Yiwen S. and Yanlin S.).

### C3. Keywords
Add "Sustainable Development" (already implied), "Textile Industry," and "China" to improve discoverability. The journal allows 3–10 keywords.

### C4. Data Availability Statement
"Avaible" contains a typo. Provide a specific repository link or DOI rather than a vague statement.

### C5. "Polycrisis" Framing
The term "polycrisis" (Tooze, 2022) is evocative but potentially overused. Consider whether the analysis actually requires this framing — the paper studies sequential shocks (WTO, 2008 GFC, 2017 environmental regulation, COVID-19), which is more accurately described as "compound shocks" or "sequential crises." The term "polycrisis" typically implies simultaneous, interacting crises. Clarify the temporal structure of shocks.

### C6. Reference Format
MDPI uses numbered references in brackets [1], [2], etc., not APA/Harvard (Author, Year). The references will need reformatting for final submission. The LaTeX template handles this automatically with the `.bib` file.

---

## D. SUMMARY OF REQUIRED ACTIONS

| Priority | Issue | Action |
|----------|-------|--------|
| **HIGH** | A1: BERTopic vs LLM scoring | Clarify actual method used; report topic keywords if BERTopic, or prompt design if LLM |
| **HIGH** | A2: Data constraint transparency | Report document length distribution; discuss impact on pre-2015 topic reliability |
| **HIGH** | A3: SCM single-donor problem | Report RMSPE; move placebo rank to main text; downgrade causal language |
| **HIGH** | A5: NLP validity diagnostics | Report document-length correlation; report external benchmark correlation |
| **MODERATE** | A4: ITS N consistency | Reconcile N=25 vs N=27; justify HAC lag |
| **MODERATE** | B1: Theoretical framework | Add microeconomic formalization sketch |
| **MODERATE** | B2: References | Add 5–8 references; include text-as-data literature |
| **MODERATE** | B3: Mechanism power | Clarify N; report adjusted R² and CIs |
| **MODERATE** | B4: Figure accuracy | Fix SDiD reference; add synergy methods |
| **MINOR** | C1–C6 | Abstract length, author names, keywords, typo, polycrisis framing, ref format |

---

## E. OVERALL RECOMMENDATION

**Major Revision.** The manuscript has genuine strengths — the case is intrinsically important, the mixed-methods design is appropriate, the conceptual framework is creative, and the honest reporting of statistical uncertainty is refreshing. However, the methodological transparency issues (A1, A2, A5) must be resolved before the contribution can be properly evaluated. I am optimistic that these issues can be addressed in a revision and look forward to seeing the improved manuscript.

---

*Review Date: 2026-06-13*
