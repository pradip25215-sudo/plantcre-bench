# PlantCRE-Bench: results so far (2026-10-22)

## Reproduction checks
- Our 6-mer ridge matches the published plantGREP 6-mer linear model (warm 0.488 vs 0.488, cold 0.422 vs 0.422).
- Published models on their own test sets: Jores promoter CNN 0.714 / 0.670 (leaf / proto),
  Gorjifard terminator CNN 0.762 / 0.672 (tobacco / maize), plantGREP CNN 0.689-0.498 (light ... maize).
- Our CNN re-implementation: 0.668 / 0.650 promoters, 0.624 / 0.549 terminators, 0.576 enhancers light.
  Weaker than the published models, most of all on terminators.

## Experiment 1: homology leakage
- 32-36% of test sequences in all three published splits have a train homolog at score >= 40 (22-28% at >= 60).
- Frozen probes (6-mer, NT-v2-100M, AgroNT-1B): random vs homology split differ by <= 0.01 R2, inside seed noise,
  on promoters, terminators and enhancers (unforced _nf splits).
- CNN on enhancers: 0.02-0.03 R2 drop from random to homology splits in all 5 conditions. Promoters/terminators:
  CNN seed-to-seed variation too large to see a drop (one t60 run collapsed to proto 0.309).
- R2 by homology bin on original test sets: the >= 100 bin is not easier. 'No hit' bin is hardest everywhere.

## Frozen embeddings vs 6-mers (original splits, best layer chosen on validation)
- AgroNT-1B layer 12 beats 6-mers on promoters (0.530 vs 0.483 leaf) and terminators (0.456 vs 0.293 tobacco),
  loses on enhancers (0.436 vs 0.496 light; 0.204 vs 0.283 maize).
- NT-v2-100M layer 10 is below 6-mers on promoters and enhancers, above on terminators tobacco.
- Best layer is the same for every task: NT-v2 layer 10 of 22, AgroNT layer 12 of 40.

## Experiment 3: TFBS-shuffle effects (741 pairs, WT and homologs never in training)
- delta Spearman, light: CNN (mean of 3 inits) 0.519, 6-mer 0.513, AgroNT 0.385, NT-v2 0.381.
- Zero-shot masked-LM score: Spearman -0.07 to 0.08, sign accuracy 0.43-0.54 (chance).
- Averaging 3 CNN inits beats every single init.
- Noise ceiling from replicates (eVal_reps.rds): max reachable delta Spearman 0.85-0.94 (min_bc 1).
  Best model reaches 42-60% of it. A single replicate predicts another better than any model
  (light 0.716 vs 0.519), except warm where 6-mer (0.433) is close to one replicate (0.459).
- Same scale (% of reachable Pearson r): WT strength 68-84% for all models (light/dark/warm/cold),
  mutation effect 31-63%. Gap of 20-35 points for every model. Maize lowest for both.

## Still to do (updated 2026-10-25)
- Formal novelty check (proposal section 8) -> docs/prior-art.md, before any writing.
- D2-filtered rerun of the final tables (controls out of the metrics).
- CNN with 5 seeds on promoters/terminators; NT-v2 fine-tuning on the enhancer _nf splits (Experiment 1).
- Results pending tonight: CNN on the same 100k rows as fine-tuning (exp3_cnn_sub100k), NT-v2 fine-tuned on terminators.
- plantGREP's own model on the variant pairs (it saw 540 of the WTs in training, report separately).
- Positional k-mer baseline for the terminator result in Experiment 2.
- Figures (4 main) and the talk with Dr. Kumar about authorship and bioRxiv.

## Experiment 2 (2026-10-25): 6-mer composition inside the embeddings
- 6-mer counts explain 44-52% of embedding variance (NT-v2 L10 and AgroNT L12, all datasets).
- R2 that the non-composition residual adds on top of 6-mers: terminators +0.09 to +0.20, promoters +0.05 to +0.10,
  enhancers +0.006 to +0.024. On enhancers the composition part alone beats the full embedding.
- Guess (not tested): 6-mer counts lose position, and terminator strength depends on PAS / cleavage position.
  Needs a positional k-mer baseline.

## Fine-tuned NT-v2-100M (2026-10-25)
- Promoters, test R2 leaf / proto: random 0.700 / 0.677, homology t40 0.700 / 0.660, t60 0.700 / 0.668.
  Frozen probe was 0.44 / 0.36. Matches the published Jores CNN. No homology drop even for this model.
- Experiment 3 (forced t40 split, train subsampled to 100k rows): delta rho light 0.534, dark 0.504, warm 0.438,
  cold 0.509, maize 0.295. Same level as the CNN trained on the full train set. Val loss rises after epoch 2.
- Comparison with the CNN is not fair yet (100k vs 270k rows). CNN on the same 100k rows: exp3_cnn_sub100k.
