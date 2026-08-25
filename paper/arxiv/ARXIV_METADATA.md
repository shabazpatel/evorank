# arXiv submission metadata (copy into the form)

**Title**
EvoRank: LLM-Guided Evolution of Multi-Objective Learning-to-Rank Pipelines

**Authors** (in order, as on the paper)
Rayhan Patel (University of Maryland), Shabaz Patel (Independent Researcher)

**Abstract**
We present EvoRank, an open autonomous ranking engineer: an LLM-guided evolutionary loop that discovers complete Learning-to-Rank pipelines (features, models, losses, ensembles) for multi-objective e-commerce search. On the Expedia ICDM 2013 dataset, with relevance, conversion, and revenue as competing objectives, three independent runs each converge within 50 iterations (about ten dollars) on interpretable pipelines that beat an Optuna-tuned LambdaMART on 60k held-out queries, an advantage that persists at full data scale and places in the top 6 percent of the original competition. A first campaign, evolving only training objectives, builds the central design rule: it appeared to work on its selection fold (the small dataset it uses to pick winners) while a transfer audit, re-scoring winners on held-out data, showed the gains were almost entirely fitness noise (the randomness of its own scoring), and neither seeded domain knowledge nor richer diagnostic feedback changed what transferred. The deciding quantity is measurable in advance: search-space headroom relative to fitness noise. We package this as a headroom gate that predicts, before any LLM spend, whether the loop will pay off, and we release the system, the auditing tools, and a catalog of failure modes with their guardrails, so teams can apply the procedure to their own ranking stacks. 

**Comments**
8 pages, 4 figures, 2 tables. Accepted at GenAIECommerce'26, the Third Workshop on Agentic and Generative AI for E-Commerce, co-located with RecSys, September 28, 2026, Minneapolis, MN, USA. Proceedings on the workshop website: https://genai-ecommerce.github.io/GenAIECommerce2026. Code and all machine-generated results: https://github.com/shabazpatel/evorank

**Primary category**
cs.IR (Information Retrieval)

**Cross-lists**
cs.LG (Machine Learning); cs.AI (Artificial Intelligence)

**License**
CC BY 4.0 (matches the copyright footnote required by the workshop)

**Journal reference / DOI**
Leave blank at submission. The workshop publishes proceedings on its website (not CEUR-WS this year); once the paper page is online, add a Comments/Journal-ref update pointing to https://genai-ecommerce.github.io/GenAIECommerce2026.

**Files in the archive** (`evorank_arxiv.tar.gz`)
main.tex, ceurart.cls, flowchart.pdf, transfer_contrast.pdf

**Checks before pressing submit**
- The arXiv build log shows pdfLaTeX (not latex+dvips) and 8 pages.
- Author list and ORCIDs match the camera-ready; link ORCIDs in the arXiv author fields.
- Title footnote and Conclusion URL resolve to the public repository.
