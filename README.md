# Low-Rank Approximation and Matrix Completion in Retail Analytics

Numerical Linear Algebra project applying SVD-based
methods to a real retail transaction dataset from Korzinka.

## What this is about

The core question I asked was simple: does customer shopping behaviour actually have a
low-rank structure? I had a full year (2025) of loyalty-card transaction data, aggregated
into a customer × product-category matrix — **300,395 customers by 498 product categories**,
with about 20.5 million non-zero spending entries (13.7% density).

If customer behaviour really is driven by a small number of underlying "shopping patterns"
rather than being essentially random, then a rank-k SVD approximation should be able to
capture most of the matrix using far fewer dimensions than the raw data. I tested this
directly, then used the resulting low-dimensional representation for a few practical things:
compressing the data, interpreting latent product-category patterns, clustering customers,
and finally trying to predict (complete) hidden spending values.

## Methods used

- **Singular Value Decomposition (SVD)** — the core tool: A ≈ UₖΣₖVₖᵀ
- **Randomized SVD** (via scikit-learn) — the matrix is too large for a full/exact SVD, so I
  compute only the leading components using randomized projection, and cross-check against
  `scipy.sparse.linalg.svds`
- **Frobenius norm / relative error** — to measure how good the rank-k approximation is,
  grounded in the Eckart–Young theorem (truncated SVD is provably optimal in this norm)
- **K-Means clustering** on the first ten SVD latent dimensions, to segment customers by
  purchasing behaviour
- **Matrix completion** — hide 10% of observed entries, train a rank-50 SVD on the rest,
  predict the hidden entries as l_i · r_j (customer latent vector dot category latent vector),
  and evaluate with RMSE/MAE against a simple category-average baseline

## What I found

- The singular values decay quickly and then flatten out — consistent with approximate
  low-rank structure. Relative Frobenius error drops from ~0.80 at rank 1 to **0.088 at
  rank 300** (i.e. under 10% error using well under the full 498 dimensions).
- The right singular vectors give interpretable latent product-category patterns —
  combinations of categories that tend to move together across customers.
- Projecting customers onto the first couple of latent dimensions shows clear, non-random
  structure, and K-Means on the first ten dimensions produces visibly separated customer
  segments.
- Matrix completion was the one place my approach came up short: the rank-50 SVD model
  (RMSE 0.120, MAE 0.058) was actually **beaten by a plain category-average baseline**
  (RMSE 0.085, MAE 0.049). So while SVD is clearly useful for compression, interpretation,
  and segmentation here, a basic truncated-SVD completion model isn't enough on its own for
  accurate spending prediction — this points at some real limitations I discuss below
  rather than a clean win.

## Why the baseline beat SVD at completion, and what I'd try next

A zero or missing entry in this matrix is ambiguous — it can mean genuine disinterest, or
just that the customer never had the chance to buy that category, or bought it somewhere
else. A plain truncated SVD doesn't distinguish between these cases, which likely explains
why the category-average baseline held up better. Some natural next steps: regularized
matrix factorization (ALS/SGD-based, which handle missingness more explicitly), testing
whether behaviour shifts across seasons/months rather than treating the whole year as static,
and comparing results with and without the ℓ2 row-normalization I used (which flatten out
differences in total spend).
