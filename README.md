# Infer-and-widen, or not?

In recent years, there has been substantial interest in the task of selective inference: inference on a parameter that is selected from the data. Many of the existing proposals fall into what we refer to as the infer-and-widen framework: they produce symmetric confidence intervals whose midpoints do not account for selection and therefore are biased; thus, the intervals must be wide enough to account for this bias. In this paper, we investigate infer-and-widen approaches in three vignettes: the winner's curse, maximal contrasts, and inference after the lasso. In each of these examples, we show that a state-of-the-art infer-and-widen proposal leads to confidence intervals that are wider than a non-infer-and-widen alternative. Furthermore, even an "oracle" infer-and-widen confidence interval -- the narrowest possible interval that could be theoretically attained via infer-and-widen -- can be wider than the alternative.

See our paper ["Infer-and-widen, or not?"](https://arxiv.org/abs/2408.06323) for further details.

## Reproducibility

`main/` contains python methods code which is locally imported. Otherwise, code for each of the paper figures is contained in the standalone script of the correspond name, e.g., code for Figure 1 panel (a) is contained in `Fig-1a...`. Appendix figures are similarly coded, except for figures A2 and A3 whose results can be generated from `Fig-2...` files.
