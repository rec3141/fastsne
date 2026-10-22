# fastsne

GPU t-SNE for very large TNF tables (nanopore reads, contigs). Single CUDA binary: needs only the CUDA toolkit (cuFFT, cuBLAS, thrust).

```
nvcc -O3 -arch=sm_70 -std=c++14 -o fastsne fastsne.cu -lcufft -lcublas      # sm_80 for A100, etc.
./fastsne -i tnf.tsv -o tsne.tsv --dedup 2
./fastsne -i @list_of_lrn_files.txt -o tsne.tsv --dedup 2                    # many files
python3 render.py tsne.tsv tsne.png --color labels.tsv --cat                 # quick PNG
```

Input rows: `id<TAB>f1<TAB>f2...` (header lines skipped). Output: `id x y` in input order.

## Timings (Quadro GV100, 16 threads)

| data | N | total | of which: kNN | optimisation (1000 it) |
|---|---|---|---|---|
| marine assembly tnf.tsv | 1.55M | 14 s | 3 s | 7 s |
| nanopore reads (5.6% of out_dana_bc) | 9.36M | 110 s | 22 s | 45 s |

Reading 20 GB of text takes about 7 s once it's in the page cache.

## Sample + project (more points than fit in memory)

```
./fastsne -i @sample.txt -o ref.tsv --dedup 2 --save-model ref.model      # embed a sample, keep the reference
./fastsne --project ref.model -i @all_files.txt -o all.tsv               # stream everything else onto it
```

Projection streams the inputs in batches (`--batch-gb`, 8 GB of text). Each point gets the same transform + PCA as the reference. It is then matched to its 16 nearest reference points through the reference's inverted file, and placed at their kernel-weighted mean position (perplexity 5). The output adds `nn_dist`, the distance to the nearest reference point, for QC.

On the marine assembly (reference = 50% sample), all 1.55M contigs projected in 3 s. The median GC difference between a contig and its 10 nearest map neighbours was:
- 0.92 for held-out contigs,
- 0.66 for contigs embedded directly,
- 9.6 for random pairs.

The model file is about 136 B per reference point (32-d PCA + 2-d position), i.e. about 4 GB for 30M.

## How it works

1. CLR transform (pseudocount = half the smallest nonzero frequency of the row, about 0.5 counts), then PCA to 32 dims.
2. kNN: k-means inverted file (N/1000 clusters, 32 probes), then one NN-descent refinement round. Recall@48 is about 0.87 at 9M.
3. Perplexity affinities on k=48 (perplexity 16).
4. Gradient descent (early exaggeration 12 for 250 iterations, then 750 more, lr = N/12).
   - Attraction uses a reverse CSR of the kNN graph, so there are no atomics.
   - Repulsion uses one FFT convolution of the gridded density with 1/(1+r²) and analytic gradient kernels (bilinear splat/gather, h=0.25). Measured error of the repulsive force is 2–3%.

## Worms (`--dedup`)

Very high coverage genomes (e.g. a Planktothrix bloom) give kNN neighbourhoods made of overlapping reads. The kNN graph is then a 1-D chain along the genome, and t-SNE draws it as worms and rings.

These reads are detectable because they share most of their neighbours with their neighbours: about 0.5 vs a median of 0.2. Global exaggeration does *not* fix the worms (they tangle into a hairball). `--dedup A` gives such reads a lower repulsion mass instead, which acts as local exaggeration, so the group collapses into a compact blob. Groups are touched only where neighbour sharing is above the median by more than `--dedup-margin`. Reasonable values for A are 1–4, with 2 a good default.

Other knobs: `--shrink G` (shrink large label-propagation groups, works for contigs, not for noisy reads), `--mass-file`, `--exag`, `--iters`, `--k`, `--pca`, `--sample F` (subsample rows on read), `--check-knn N`, `--check-grad N`, `--diag FILE`, `--groups FILE`.
